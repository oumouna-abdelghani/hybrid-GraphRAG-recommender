"""
Synchronous wrapper around neo4j-agent-memory (0.7.0, Bolt backend).

DATA SHAPES
  profile : {"like":    {"person": {...}, "genre": {...}, "decade": {...}},
             "dislike": {same}}
            Stored as one library Preference per item:
              category   = "<polarity>_<kind>"   e.g. "like_genre"
              preference = value                 e.g. "Sci-Fi"

main.py is synchronous and the library is async, so this class owns ONE private
event loop and exposes plain blocking methods. Memory is optional: a failure
prints one warning and the methods return safe defaults, so search keeps working.
"""

import asyncio
import os
import time
import uuid

try:
    from neo4j_agent_memory import MemoryClient, MemorySettings
    from neo4j_agent_memory.schema.models import TraceOutcome
except ImportError:  # library not installed -> run without memory
    MemoryClient = None

POLARITIES = ("like", "dislike")


def build_settings():
    """Same Neo4j database as main.py; no LLM, no entity extraction."""
    return MemorySettings(
        backend="bolt",
        neo4j={
            "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            "username": os.getenv("NEO4J_USER", "neo4j"),
            "password": os.getenv("NEO4J_PASSWORD", ""),
            "database": os.getenv("NEO4J_DATABASE", "neo4j"),
        },
        embedding=os.getenv("MEMORY_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        llm=None,
        extraction={"extractor_type": "none"},
        resolution={"strategy": "none"},
        geocoding={"enabled": False},
        enrichment={"enabled": False},
    )


class MemoryManager:
    def __init__(self, user_id=None):
        self.user_id = user_id or os.getenv("MEMORY_USER_ID", "default-user")
        self.session_id = None
        self.ready = False
        self._client = None
        self._warned = False
        self._loop = asyncio.new_event_loop()

    # ---------- plumbing ----------

    def _call(self, coro, default=None):
        try:
            return self._loop.run_until_complete(coro)
        except Exception as error:
            if not self._warned:
                print(f"[memory] {type(error).__name__}: {error}")
                self._warned = True
            return default

    def start(self):
        """Connect and open a fresh CLI session. Returns True when usable."""
        if os.getenv("MEMORY_ENABLED", "true").lower() in {"0", "false", "no"}:
            print("Memory is disabled (MEMORY_ENABLED=false).")
            return False

        if MemoryClient is None:
            print("neo4j-agent-memory is not installed; running without memory.")
            return False

        async def connect():
            client = MemoryClient(build_settings())
            await client.__aenter__()
            return client

        self._client = self._call(connect())

        if self._client is None:
            print("Continuing without memory.")
            return False

        self.ready = True
        self.session_id = f"cli-{uuid.uuid4().hex[:8]}"
        print(f"Memory connected (session {self.session_id}, user {self.user_id}).")
        return True

    def close(self):
        if self.ready:
            self._call(self._client.__aexit__(None, None, None))
            self.ready = False
        self._loop.close()

    # ---------- short-term memory ----------

    def add_message(self, role, content):
        if not self.ready:
            return

        self._call(
            self._client.short_term.add_message(
                self.session_id,
                role,
                content,
                user_identifier=self.user_id,
                extract_entities=False,
            )
        )

    def context(self, query):
        """Conversation memory as text for an LLM prompt (get_context, else last turns)."""
        if not self.ready:
            return ""

        async def fetch():
            try:
                text = await self._client.get_context(
                    query, session_id=self.session_id
                )
            except Exception:
                text = ""

            if text:
                return str(text)

            conversation = await self._client.short_term.get_conversation(
                self.session_id
            )
            return "\n".join(
                f"{message.role.value}: {message.content}"
                for message in conversation.messages[-8:]
            )

        return self._call(fetch(), "")

    # ---------- long-term memory ----------

    def save_preference(self, polarity, kind, value):
        """Store one preference. A flipped opinion supersedes the old one.
        Returns True if it was new."""
        if not self.ready:
            return False

        async def save():
            memory = self._client.long_term
            existing = await memory.get_preferences_for(self.user_id)

            category = f"{polarity}_{kind}"
            opposite = f"{'dislike' if polarity == 'like' else 'like'}_{kind}"

            def matches(pref, wanted_category):
                return (
                    pref.category == wanted_category
                    and pref.preference.lower() == value.lower()
                )

            if any(matches(pref, category) for pref in existing):
                return False

            new = await memory.add_preference(
                category,
                value,
                user_identifier=self.user_id,
                generate_embedding=False,
                context="Stated by the user in the CLI",
            )

            for old in existing:
                if matches(old, opposite):
                    await memory.supersede_preference(old.id, new.id)

            return True

        return self._call(save(), False)

    def load_profile(self):
        """Active preferences as a profile dict (see DATA SHAPES)."""
        profile = {side: {} for side in POLARITIES}

        if not self.ready:
            return profile

        prefs = self._call(
            self._client.long_term.get_preferences_for(self.user_id), []
        )

        for pref in prefs:
            side, _, kind = pref.category.partition("_")

            if side in profile and kind:
                profile[side].setdefault(kind, set()).add(pref.preference)

        return profile

    # ---------- reasoning memory ----------

    def trace(self, task):
        return Trace(self, task)


class Trace:
    """`with memory.trace(task) as t:` -> start_trace ... complete_trace.
    The trace is always completed, success or failure."""

    def __init__(self, memory, task):
        self.memory = memory
        self.task = task
        self.trace_id = None
        self.summary = ""

    def __enter__(self):
        memory = self.memory

        if memory.ready:
            trace = memory._call(
                memory._client.reasoning.start_trace(memory.session_id, self.task)
            )
            self.trace_id = trace.id if trace else None

        return self

    def step(self, action, arguments, result, started):
        """One pipeline stage: add_step + record_tool_call.
        `started` is a time.perf_counter() value taken before the stage."""
        if self.trace_id is None:
            return

        duration_ms = int((time.perf_counter() - started) * 1000)
        reasoning = self.memory._client.reasoning

        async def record():
            step = await reasoning.add_step(self.trace_id, action=action)
            await reasoning.record_tool_call(
                step.id,
                action,
                arguments,
                result=result,
                duration_ms=duration_ms,
            )

        self.memory._call(record())

    def __exit__(self, exc_type, exc, tb):
        if self.trace_id is not None:
            failure = f"{exc_type.__name__}: {exc}" if exc_type else ""
            outcome = TraceOutcome(
                success=exc_type is None,
                summary=self.summary or failure or "completed",
            )
            self.memory._call(
                self.memory._client.reasoning.complete_trace(
                    self.trace_id, outcome=outcome
                )
            )

        return False