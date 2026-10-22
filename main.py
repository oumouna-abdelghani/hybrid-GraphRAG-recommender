import hashlib
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"

# Only used by the optional "explain these picks" step.
GENERATION_MODEL = os.getenv(
    "GEMINI_GENERATION_MODEL",
    "gemini-3.5-flash-lite",
)

# Keep paths relative to this Python file, not the
# terminal's current working directory.
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "tv_movies_graphrag_dataset.csv"
DB_PATH = BASE_DIR / ".chroma_tv_recommender"

COLLECTION_NAME = "tv_movies"
BATCH_SIZE = 50

# Neo4j connection settings come from the .env file.
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

# How CSV columns become graph relationships.
#   to_title   -> (node)-[:TYPE]->(:Title)   e.g. (Person)-[:ACTED_IN]->(Title)
#   from_title -> (:Title)-[:TYPE]->(node)   e.g. (Title)-[:IN_GENRE]->(Genre)
# Adding a new kind of relationship later = adding one dict here.
RELATIONS = [
    {"column": "cast", "label": "Person", "type": "ACTED_IN", "direction": "to_title"},
    {"column": "directors", "label": "Person", "type": "DIRECTED", "direction": "to_title"},
    {"column": "creators", "label": "Person", "type": "CREATED", "direction": "to_title"},
    {"column": "producers", "label": "Person", "type": "PRODUCED", "direction": "to_title"},
    {"column": "genres", "label": "Genre", "type": "IN_GENRE", "direction": "from_title"},
    {"column": "franchise", "label": "Franchise", "type": "PART_OF", "direction": "from_title"},
]

# Columns added for the graph. Older CSV files without them still load.
OPTIONAL_COLUMNS = ["directors", "creators", "cast", "producers", "franchise"]

# Hybrid ranking knobs: final score = semantic similarity + graph bonus.
CANDIDATE_POOL = 20       # how many vector candidates to fuse before cutting to top_k
GRAPH_BOOST = 0.05        # bonus per unit of connection weight
MAX_GRAPH_BONUS = 0.25    # cap, so the graph can never fully override meaning

# How strongly each kind of shared connection says "these two titles are related".
CONNECTION_WEIGHTS = {
    "FRANCHISE": 2.0,
    "DIRECTED": 1.5,
    "CREATED": 1.5,
    "ACTED_IN": 1.0,
    "PRODUCED": 0.5,
}

ROLE_LABELS = {
    "FRANCHISE": "same franchise",
    "DIRECTED": "director",
    "CREATED": "creator",
    "ACTED_IN": "cast",
    "PRODUCED": "producer",
}

NAME_SUFFIXES = {"jr", "sr"}


# --------------------------------------------------
# 2. INITIALIZE THE TWO STORES
# --------------------------------------------------

# Store #1: vector database (meaning). Unchanged.
# Unlike an in-memory client, PersistentClient saves
# the database to disk and loads it again next time.
chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# Store #2: graph database (relationships).
# GraphDatabase.driver() does not connect yet; the first query does.
# main() checks the connection and prints a friendly error if it fails.
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)


# --------------------------------------------------
# 3. PREPARE ENRICHED TEXT FOR EACH TITLE
# --------------------------------------------------

def clean_value(value):
    """Convert missing CSV values into empty strings."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def split_list(value):
    """Turn 'A; B; C' from a CSV cell into ['A', 'B', 'C']."""
    return [
        part.strip()
        for part in clean_value(value).split(";")
        if part.strip()
    ]


def create_embedding_text(item):
    """
    Combine useful fields into one enriched document.

    The embedding model can now consider the plot, genres,
    themes, mood, media type, release year, and the people
    and franchise connected to the title.
    """

    parts = [
        f"Title: {clean_value(item.get('title'))}.",
        f"Media type: {clean_value(item.get('type'))}.",
        f"Description: {clean_value(item.get('description'))}.",
        f"Genres: {clean_value(item.get('genres'))}.",
        f"Themes: {clean_value(item.get('themes'))}.",
        f"Mood: {clean_value(item.get('mood'))}.",
        f"Release year: {clean_value(item.get('release_year'))}.",
    ]

    for label, column in [
        ("Directed by", "directors"),
        ("Created by", "creators"),
        ("Cast", "cast"),
        ("Franchise", "franchise"),
    ]:
        names = ", ".join(split_list(item.get(column)))

        if names:
            parts.append(f"{label}: {names}.")

    return " ".join(parts)


def create_embeddings(texts, task_type):
    """Generate Gemini embeddings for a batch of texts."""

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type
        ),
    )

    return [
        embedding.values
        for embedding in response.embeddings
    ]


def create_text_hash(text):
    """
    Create a fingerprint of the prepared document.

    The embedding model is part of the fingerprint, so switching
    models automatically re-embeds everything instead of mixing
    vectors from two different models.
    """
    return hashlib.sha256(
        f"{EMBEDDING_MODEL}|{text}".encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# 4. SYNCHRONIZE THE CSV WITH BOTH STORES
# --------------------------------------------------

def load_catalog():
    """Read and validate the CSV. Both stores are built from this."""

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Place the CSV beside main.py."
        )

    df = pd.read_csv(CSV_PATH, encoding="utf-8")

    required_columns = {
        "id",
        "title",
        "description",
        "type",
        "genres",
        "themes",
        "mood",
        "release_year",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            "Missing CSV columns: "
            + ", ".join(sorted(missing_columns))
        )

    if df["id"].isna().any():
        raise ValueError("Every title must have an ID.")

    if df["id"].astype(str).duplicated().any():
        raise ValueError("Every title must have a unique ID.")

    if df["title"].isna().any():
        raise ValueError("Every title must have a title.")

    for column in OPTIONAL_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    return df


def sync_vector_store(df):
    """
    Add new titles, update changed titles, and remove
    titles that no longer exist in the CSV (ChromaDB side).

    Unchanged documents do not need new embeddings, because
    every embedding costs an API request.
    """

    # Read existing IDs and their saved fingerprints.
    existing = collection.get(
        include=["metadatas"]
    )

    existing_hashes = {}

    for record_id, metadata in zip(
        existing["ids"],
        existing["metadatas"],
    ):
        existing_hashes[record_id] = metadata.get(
            "text_hash", ""
        )

    current_ids = set()
    pending = []

    for _, row in df.iterrows():
        item = row.to_dict()

        # ChromaDB IDs must be strings.
        record_id = str(item["id"])
        current_ids.add(record_id)

        document = create_embedding_text(item)
        text_hash = create_text_hash(document)

        # If the document hasn't changed, reuse its
        # existing embedding without an API request.
        if existing_hashes.get(record_id) == text_hash:
            continue

        # Build metadata from supported scalar values.
        # title_id duplicates the record ID on purpose: Chroma can
        # filter on metadata, which lets the graph hand us a list
        # of allowed titles ({"title_id": {"$in": [...]}}).
        metadata = {
            "title_id": record_id,
            "title": clean_value(item.get("title")),
            "type": clean_value(item.get("type")),
            "genres": clean_value(item.get("genres")),
            "text_hash": text_hash,
        }

        year = pd.to_numeric(
            item.get("release_year"),
            errors="coerce",
        )

        if pd.notna(year):
            metadata["release_year"] = int(year)

        pending.append({
            "id": record_id,
            "document": document,
            "metadata": metadata,
        })

    # Remove database entries deleted from the CSV.
    stale_ids = set(existing["ids"]) - current_ids

    if stale_ids:
        collection.delete(ids=list(stale_ids))
        print(f"Removed {len(stale_ids)} deleted titles.")

    if not pending:
        print(
            f"Vector store is up to date: "
            f"{collection.count()} titles."
        )
        print("No catalog embedding requests needed.")
        return

    print(
        f"Generating embeddings for "
        f"{len(pending)} new or changed titles..."
    )

    # Process the catalog in batches rather than
    # sending the entire dataset in one huge request.
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start:start + BATCH_SIZE]

        documents = [
            item["document"] for item in batch
        ]

        embeddings = create_embeddings(
            documents,
            task_type="RETRIEVAL_DOCUMENT",
        )

        # Upsert inserts new records and updates existing
        # ones using their stable IDs.
        collection.upsert(
            ids=[item["id"] for item in batch],
            embeddings=embeddings,
            documents=documents,
            metadatas=[
                item["metadata"] for item in batch
            ],
        )

        print(
            f"Saved {min(start + len(batch), len(pending))}"
            f"/{len(pending)} changed titles."
        )

    print(
        f"Vector store ready: {collection.count()} titles."
    )


# Graph schema: one uniqueness rule per node type. This also makes
# MERGE fast and guarantees there is exactly one node per person.
GRAPH_CONSTRAINTS = [
    "CREATE CONSTRAINT title_id IF NOT EXISTS "
    "FOR (t:Title) REQUIRE t.id IS UNIQUE",
    "CREATE CONSTRAINT person_name IF NOT EXISTS "
    "FOR (p:Person) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT genre_name IF NOT EXISTS "
    "FOR (g:Genre) REQUIRE g.name IS UNIQUE",
    "CREATE CONSTRAINT franchise_name IF NOT EXISTS "
    "FOR (f:Franchise) REQUIRE f.name IS UNIQUE",
]

WIPE_GRAPH = (
    "MATCH (n) "
    "WHERE n:Title OR n:Person OR n:Genre OR n:Franchise "
    "DETACH DELETE n"
)

CREATE_TITLES = """
UNWIND $rows AS row
CREATE (:Title {
    id: row.id,
    title: row.title,
    type: row.type,
    description: row.description,
    release_year: row.release_year
})
"""


def build_link_query(relation):
    """
    Build the Cypher that connects titles to one kind of node.

    Labels and relationship types cannot be passed as query
    parameters in Cypher, so they are inserted with an f-string.
    That is safe here because they come from the RELATIONS
    constant above, never from user input.
    """

    label = relation["label"]
    rel_type = relation["type"]

    if relation["direction"] == "to_title":
        pattern = f"(n)-[r:{rel_type}]->(t)"
    else:
        pattern = f"(t)-[r:{rel_type}]->(n)"

    return f"""
UNWIND $pairs AS pair
MATCH (t:Title {{id: pair.title_id}})
MERGE (n:{label} {{name: pair.name}})
MERGE {pattern}
SET r.order = pair.order
"""


def build_graph_rows(df):
    """Turn each CSV row into a plain dict the graph loader understands."""

    rows = []

    for _, row in df.iterrows():
        item = row.to_dict()

        year = pd.to_numeric(
            item.get("release_year"),
            errors="coerce",
        )

        rows.append({
            "id": str(item["id"]),
            "title": clean_value(item.get("title")),
            "type": clean_value(item.get("type")),
            "description": clean_value(item.get("description")),
            "release_year": int(year) if pd.notna(year) else None,
            "links": {
                relation["column"]: split_list(
                    item.get(relation["column"])
                )
                for relation in RELATIONS
            },
        })

    return rows


def load_graph(tx, rows):
    """
    Rebuild the whole graph from the CSV inside ONE transaction.

    If anything fails, Neo4j rolls everything back and the old
    graph stays intact.
    """

    tx.run(WIPE_GRAPH).consume()
    tx.run(CREATE_TITLES, rows=rows).consume()

    for relation in RELATIONS:
        # "order" keeps the billing order of the cast
        # (0 = top billed) as a property on the relationship.
        pairs = [
            {
                "title_id": row["id"],
                "name": name,
                "order": position,
            }
            for row in rows
            for position, name in enumerate(
                row["links"][relation["column"]]
            )
        ]

        if pairs:
            tx.run(
                build_link_query(relation),
                pairs=pairs,
            ).consume()


def sync_graph(df):
    """
    Rebuild the Neo4j graph from the CSV.

    Unlike embeddings, graph writes cost no API calls and take
    milliseconds at this size, so a clean rebuild is simpler and
    safer than diffing. The CSV stays the single source of truth.
    """

    rows = build_graph_rows(df)

    for statement in GRAPH_CONSTRAINTS:
        driver.execute_query(statement)

    with driver.session() as session:
        session.execute_write(load_graph, rows)

    records, _, _ = driver.execute_query(
        "RETURN COUNT { (:Title) } AS titles, "
        "COUNT { (:Person) } AS people, "
        "COUNT { ()-[]->() } AS relationships"
    )

    summary = records[0]

    print(
        f"Graph ready: {summary['titles']} titles, "
        f"{summary['people']} people, "
        f"{summary['relationships']} relationships."
    )


# --------------------------------------------------
# 5. RECOGNIZE PEOPLE AND TITLES IN A QUERY
# --------------------------------------------------

# Names that exist in the graph, loaded once after the sync.
ENTITY_INDEX = {
    "people_full": {},   # "christopher nolan" -> "Christopher Nolan"
    "people_last": {},   # "nolan"             -> "Christopher Nolan"
    "titles": {},        # "inception"         -> ("125", "Inception")
}

SEED_PATTERN = re.compile(
    r"\b(?:like|similar to|same as|reminds? me of|in the style of)\s+(.+)",
    re.IGNORECASE,
)


def fold(text):
    """Lowercase, drop accents and punctuation: 'Úrsula Corberó' -> 'ursula corbero'."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(
        char for char in decomposed
        if not unicodedata.combining(char)
    )
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


def load_entity_index():
    """Load every person and title name from the graph into memory."""

    people, _, _ = driver.execute_query(
        "MATCH (p:Person) RETURN p.name AS name"
    )
    titles, _, _ = driver.execute_query(
        "MATCH (t:Title) RETURN t.id AS id, t.title AS title"
    )

    ENTITY_INDEX["people_full"] = {
        fold(record["name"]): record["name"]
        for record in people
    }

    # A last name alone ("Nolan") is only usable if exactly one
    # person has it, otherwise we would guess the wrong person.
    by_last_name = defaultdict(set)

    for record in people:
        words = [
            word for word in fold(record["name"]).split()
            if word not in NAME_SUFFIXES
        ]

        if words:
            by_last_name[words[-1]].add(record["name"])

    ENTITY_INDEX["people_last"] = {
        last_name: next(iter(names))
        for last_name, names in by_last_name.items()
        if len(names) == 1 and len(last_name) >= 4
    }

    ENTITY_INDEX["titles"] = {
        fold(record["title"]): (record["id"], record["title"])
        for record in titles
    }


def find_people(query):
    """
    Find people mentioned in the query.

    - A full name matches anywhere, in any capitalization.
    - A last name alone ("Nolan") only matches if it is capitalized
      in the query, so "strong female lead" does not match Mark Strong.
    """

    folded = fold(query)

    capitalized = {
        fold(word)
        for word in re.findall(r"\w+", query)
        if word[0].isupper()
    }

    found = []

    for alias, name in ENTITY_INDEX["people_full"].items():
        if re.search(rf"\b{re.escape(alias)}\b", folded):
            found.append(name)

    for last_name, name in ENTITY_INDEX["people_last"].items():
        if name not in found and last_name in capitalized:
            found.append(name)

    return found


def find_seed_titles(query):
    """
    Find titles the user wants "more like".

    Titles are only recognized after phrases such as "like" or
    "similar to", or when wrapped in double quotes. Otherwise a
    query like "dark thriller" would match the show "Dark".
    """

    segments = re.findall(r'"([^"]+)"', query)

    match = SEED_PATTERN.search(query)

    if match:
        segments.append(match.group(1))

    found = []

    # Longest titles first, so "Spider-Man 2" wins over "Spider-Man".
    ordered_titles = sorted(
        ENTITY_INDEX["titles"].items(),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )

    for segment in segments:
        remaining = fold(segment)

        for title_key, (title_id, title) in ordered_titles:
            pattern = rf"\b{re.escape(title_key)}\b"

            if re.search(pattern, remaining):
                found.append((title_id, title))
                remaining = re.sub(pattern, " ", remaining)

    return list(dict.fromkeys(found))


def find_entities(query):
    """Return (people, seed_titles) found in the query."""
    return find_people(query), find_seed_titles(query)


# --------------------------------------------------
# 6. GRAPH LOOKUPS
# --------------------------------------------------

PEOPLE_LINKS_QUERY = """
MATCH (p:Person)-[r]->(t:Title)
WHERE p.name IN $names
RETURN t.id AS id, p.name AS person, type(r) AS role
"""

# Titles that share a person or a franchise with the seed title.
# Genres are deliberately ignored: nearly everything shares one,
# and the vector side already handles "same vibe".
NEIGHBOR_QUERY = """
MATCH (seed:Title {id: $seed_id})<-[r1]-(p:Person)-[r2]->(other:Title)
WHERE other <> seed
RETURN other.id AS id,
       p.name AS via,
       collect(DISTINCT type(r1)) AS seed_roles,
       collect(DISTINCT type(r2)) AS other_roles
UNION
MATCH (seed:Title {id: $seed_id})-[:PART_OF]->(f:Franchise)<-[:PART_OF]-(other:Title)
WHERE other <> seed
RETURN other.id AS id,
       f.name AS via,
       ['FRANCHISE'] AS seed_roles,
       ['FRANCHISE'] AS other_roles
"""

CONTEXT_QUERY = """
MATCH (t:Title)
WHERE t.id IN $ids
RETURN t.id AS id,
       t.description AS description,
       [(d:Person)-[:DIRECTED]->(t) | d.name] AS directors,
       [(c:Person)-[:CREATED]->(t) | c.name] AS creators,
       [(a:Person)-[r:ACTED_IN]->(t) | {name: a.name, order: r.order}] AS cast,
       [(t)-[:PART_OF]->(f:Franchise) | f.name] AS franchises
"""


def titles_for_people(names):
    """
    Titles connected to the named people, in any role.

    Returns (allowed_ids, reasons, matched_all). If no title links
    every named person, it falls back to titles linked to any of them.
    """

    records, _, _ = driver.execute_query(
        PEOPLE_LINKS_QUERY,
        parameters_={"names": names},
    )

    roles = defaultdict(lambda: defaultdict(set))

    for record in records:
        roles[record["id"]][record["person"]].add(record["role"])

    linked_to_everyone = {
        title_id
        for title_id, people in roles.items()
        if len(people) == len(names)
    }

    allowed_ids = linked_to_everyone or set(roles)

    reasons = {
        title_id: [
            f"{person} ({'/'.join(sorted(ROLE_LABELS[role] for role in person_roles))})"
            for person, person_roles in roles[title_id].items()
        ]
        for title_id in allowed_ids
    }

    return allowed_ids, reasons, bool(linked_to_everyone)


def connection_weight(seed_roles, other_roles):
    """A link is only as strong as the weaker of the two roles."""
    return max(
        min(CONNECTION_WEIGHTS[seed_role], CONNECTION_WEIGHTS[other_role])
        for seed_role in seed_roles
        for other_role in other_roles
    )


def describe_link(via, seed_title, seed_roles, other_roles):
    """Human-readable reason for one graph connection."""

    if seed_roles == ["FRANCHISE"]:
        return f"same franchise as {seed_title} ({via})"

    mine = "/".join(sorted(ROLE_LABELS[role] for role in seed_roles))
    theirs = "/".join(sorted(ROLE_LABELS[role] for role in other_roles))

    if mine == theirs:
        return f"{via} ({mine} in both)"

    return f"{via} ({mine} of {seed_title}, {theirs} here)"


def neighbors_of(seed_id, seed_title):
    """Titles connected to a seed title, with a score and the reasons why."""

    records, _, _ = driver.execute_query(
        NEIGHBOR_QUERY,
        parameters_={"seed_id": seed_id},
    )

    found = {}

    for record in records:
        entry = found.setdefault(
            record["id"],
            {"score": 0.0, "reasons": []},
        )

        entry["score"] += connection_weight(
            record["seed_roles"],
            record["other_roles"],
        )

        entry["reasons"].append(
            describe_link(
                record["via"],
                seed_title,
                record["seed_roles"],
                record["other_roles"],
            )
        )

    return found


def graph_context(ids):
    """Fetch directors, creators, cast, franchise, and plot for display."""

    if not ids:
        return {}

    records, _, _ = driver.execute_query(
        CONTEXT_QUERY,
        parameters_={"ids": list(ids)},
    )

    context = {}

    for record in records:
        cast = sorted(
            record["cast"],
            key=lambda member: (
                member["order"] if member["order"] is not None else 999
            ),
        )

        context[record["id"]] = {
            "description": record["description"] or "",
            "directors": record["directors"],
            "creators": record["creators"],
            "cast": [member["name"] for member in cast[:6]],
            "franchises": record["franchises"],
        }

    return context


# --------------------------------------------------
# 7. VECTOR SEARCH AND HYBRID RANKING
# --------------------------------------------------

def build_where(media_type=None, era=None, allowed_ids=None):
    """Build the ChromaDB metadata filter (user filters + graph filter)."""

    filters = []

    if media_type:
        filters.append({
            "type": {"$eq": media_type}
        })

    if era == "old":
        filters.append({
            "release_year": {"$lt": 2010}
        })
    elif era == "recent":
        filters.append({
            "release_year": {"$gte": 2020}
        })

    if allowed_ids is not None:
        filters.append({
            "title_id": {"$in": sorted(allowed_ids)}
        })

    if not filters:
        return None

    if len(filters) == 1:
        return filters[0]

    return {"$and": filters}


def vector_search(query_embedding, n_results, where=None):
    """Nearest-neighbor search in ChromaDB. Returns a list of hit dicts."""

    total = collection.count()

    if total == 0 or n_results < 1:
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, total),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []

    for record_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "id": record_id,
            "document": document,
            "metadata": metadata,
            # With cosine distance, similarity is 1 - distance.
            "similarity": 1 - distance,
        })

    return hits


def hybrid_search(query, top_k=5, media_type=None, era=None):
    """
    Combine both stores.

    1. Recognize people and "like <title>" mentions with the graph.
    2. People are a HARD constraint ("movies with Tom Hardy"):
       the vector search only looks at titles linked to them.
    3. "Like <title>" is a SOFT signal: titles sharing people or a
       franchise with it get a bonus on top of their semantic score.
    4. Rank by similarity + bonus and attach graph context.
    """

    if not query.strip():
        print("Please enter a search query.")
        return []

    if collection.count() == 0:
        print("The database is empty.")
        return []

    people, seeds = find_entities(query)

    print(
        "Graph entities -> "
        f"people: {', '.join(people) or '-'} | "
        f"similar to: {', '.join(title for _, title in seeds) or '-'}"
    )

    allowed_ids = None
    person_reasons = {}

    if people:
        allowed_ids, person_reasons, matched_all = titles_for_people(people)

        if not allowed_ids:
            print("No titles are linked to those people.")
            return []

        if len(people) > 1 and not matched_all:
            print(
                "No single title links all of them; "
                "showing titles linked to any of them."
            )

    boosts = defaultdict(float)
    neighbor_reasons = defaultdict(list)

    for seed_id, seed_title in seeds:
        for title_id, link in neighbors_of(seed_id, seed_title).items():
            boosts[title_id] += link["score"]
            neighbor_reasons[title_id].extend(link["reasons"])

    # One embedding request, reused for every vector query below.
    query_embedding = create_embeddings(
        [query],
        task_type="RETRIEVAL_QUERY",
    )[0]

    pool_size = max(CANDIDATE_POOL, top_k * 2)

    pool = {}

    for hit in vector_search(
        query_embedding,
        pool_size,
        build_where(media_type, era, allowed_ids),
    ):
        pool[hit["id"]] = hit

    # Make sure graph neighbors compete even when their text
    # is not among the closest vectors.
    if boosts:
        neighbor_ids = set(boosts)

        if allowed_ids is not None:
            neighbor_ids &= allowed_ids

        if neighbor_ids:
            for hit in vector_search(
                query_embedding,
                len(neighbor_ids),
                build_where(media_type, era, neighbor_ids),
            ):
                pool.setdefault(hit["id"], hit)

    # "Movies like Inception" should not return Inception itself.
    seed_ids = {seed_id for seed_id, _ in seeds}

    ranked = []

    for hit in pool.values():
        if hit["id"] in seed_ids:
            continue

        bonus = min(
            MAX_GRAPH_BONUS,
            GRAPH_BOOST * boosts.get(hit["id"], 0.0),
        )

        hit["graph_bonus"] = bonus
        hit["score"] = hit["similarity"] + bonus
        hit["reasons"] = (
            person_reasons.get(hit["id"], [])
            + neighbor_reasons.get(hit["id"], [])
        )

        ranked.append(hit)

    ranked.sort(key=lambda hit: hit["score"], reverse=True)
    ranked = ranked[:top_k]

    context = graph_context([hit["id"] for hit in ranked])

    for hit in ranked:
        hit["graph"] = context.get(hit["id"], {})

    return ranked


def print_results(query, hits):
    """Print ranked hits with their graph context."""

    if not hits:
        print("No matching titles found.")
        return

    print(f"\nSearch: {query}")
    print("-" * 60)

    for hit in hits:
        metadata = hit["metadata"]
        graph = hit["graph"]

        print(
            f"\n[{hit['score']:.3f}] "
            f"{metadata.get('title', 'Unknown title')}  "
            f"(semantic {hit['similarity']:.3f} "
            f"+ graph {hit['graph_bonus']:.3f})"
        )

        print(
            f"Type: {metadata.get('type', 'Unknown')} | "
            f"Year: {metadata.get('release_year', 'Unknown')}"
        )

        print(f"Genres: {metadata.get('genres', 'Unknown')}")

        if graph.get("directors"):
            print(f"Directed by: {', '.join(graph['directors'])}")

        if graph.get("creators"):
            print(f"Created by: {', '.join(graph['creators'])}")

        if graph.get("cast"):
            print(f"Cast: {', '.join(graph['cast'])}")

        if graph.get("franchises"):
            print(f"Franchise: {', '.join(graph['franchises'])}")

        if hit["reasons"]:
            print(f"Why it matched: {'; '.join(hit['reasons'])}")

        if graph.get("description"):
            print(f"Plot: {graph['description']}")


# --------------------------------------------------
# 8. OPTIONAL: LET GEMINI EXPLAIN THE PICKS (the "G" in RAG)
# --------------------------------------------------

def explain_results(query, hits):
    """
    Ask Gemini to explain the results using ONLY the retrieved
    text and graph facts. Retrieval is what makes the answer
    grounded; generation just puts it into words.
    """

    blocks = []

    for hit in hits:
        metadata = hit["metadata"]
        graph = hit["graph"]

        makers = graph.get("directors", []) + graph.get("creators", [])

        blocks.append(
            f"- {metadata.get('title')} "
            f"({metadata.get('type')}, {metadata.get('release_year', 'n/a')})\n"
            f"  Plot: {graph.get('description') or 'n/a'}\n"
            f"  Directors/creators: {', '.join(makers) or 'n/a'}\n"
            f"  Cast: {', '.join(graph.get('cast', [])) or 'n/a'}\n"
            f"  Graph links: {'; '.join(hit['reasons']) or 'none (semantic match only)'}"
        )

    prompt = (
        "You are a movie and TV recommendation assistant.\n"
        "Using ONLY the facts below, explain in 3 to 5 short sentences "
        "why these titles fit the request. Mention shared people or "
        "franchises when they appear under 'Graph links'. "
        "Do not invent plot details, cast members, or connections.\n\n"
        f"Request: {query}\n\n"
        "Results:\n" + "\n".join(blocks)
    )

    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )
    except errors.APIError as error:
        print(f"Could not generate an explanation: {error}")
        return

    print("\n" + (response.text or "").strip())


# --------------------------------------------------
# 9. INTERACTIVE TERMINAL
# --------------------------------------------------

def run_terminal():
    print("\nTV & Movie Hybrid Search (vectors + graph)")
    print("Try: 'movies with Leonardo DiCaprio', 'Nolan sci-fi',")
    print("     'something like Inception', 'a heist with clever twists'")
    print("Type 'exit' as a query to quit.")

    while True:
        query = input(
            "\nWhat would you like to watch? "
        ).strip()

        if query.lower() == "exit":
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nMedia type:")
        print("1. Any")
        print("2. TV shows")
        print("3. Movies")

        type_choice = input("Choose 1-3 [1]: ").strip()

        media_type = {
            "2": "TV Show",
            "3": "Movie",
        }.get(type_choice)

        print("\nRelease period:")
        print("1. Any")
        print("2. Older than 2010")
        print("3. 2020 or newer")

        era_choice = input("Choose 1-3 [1]: ").strip()

        era = {
            "2": "old",
            "3": "recent",
        }.get(era_choice)

        top_k_input = input(
            "Number of results [5]: "
        ).strip()

        try:
            top_k = int(top_k_input) if top_k_input else 5
            if top_k < 1:
                raise ValueError
        except ValueError:
            print("Enter a positive whole number.")
            continue

        hits = hybrid_search(
            query=query,
            top_k=top_k,
            media_type=media_type,
            era=era,
        )

        print_results(query, hits)

        if hits:
            answer = input(
                "\nExplain these picks with Gemini? [y/N]: "
            ).strip().lower()

            if answer == "y":
                explain_results(query, hits)


def main():
    if not NEO4J_PASSWORD:
        raise SystemExit(
            "NEO4J_PASSWORD is missing. "
            "Add it to your .env file (see .env.example)."
        )

    try:
        driver.verify_connectivity()
    except (ServiceUnavailable, AuthError) as error:
        raise SystemExit(
            f"Could not connect to Neo4j at {NEO4J_URI}: {error}\n"
            "Is the database running? Try: docker compose up -d"
        ) from error

    try:
        catalog = load_catalog()

        # Embeddings are expensive, so Chroma syncs incrementally.
        sync_vector_store(catalog)

        # Graph writes are cheap, so Neo4j is rebuilt from the CSV.
        sync_graph(catalog)

        load_entity_index()
        run_terminal()
    finally:
        driver.close()


if __name__ == "__main__":
    main()
