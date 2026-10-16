
import hashlib
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

load_dotenv()

client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"

# Keep paths relative to this Python file, not the
# terminal's current working directory.
BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "tv_movies_semantic_search_dataset.csv"
DB_PATH = BASE_DIR / ".chroma_tv_recommender"

COLLECTION_NAME = "tv_movies"
BATCH_SIZE = 50


# --------------------------------------------------
# 2. INITIALIZE THE PERSISTENT VECTOR DATABASE
# --------------------------------------------------

# Unlike an in-memory client, PersistentClient saves
# the database to disk and loads it again next time.
chroma_client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


# --------------------------------------------------
# 3. PREPARE ENRICHED TEXT FOR EACH TITLE
# --------------------------------------------------

def clean_value(value):
    """Convert missing CSV values into empty strings."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def create_embedding_text(item):
    """
    Combine useful fields into one enriched document.

    The embedding model can now consider the plot,
    genres, themes, mood, media type, and release year.
    """

    return (
        f"Title: {clean_value(item.get('title'))}. "
        f"Media type: {clean_value(item.get('type'))}. "
        f"Description: {clean_value(item.get('description'))}. "
        f"Genres: {clean_value(item.get('genres'))}. "
        f"Themes: {clean_value(item.get('themes'))}. "
        f"Mood: {clean_value(item.get('mood'))}. "
        f"Release year: {clean_value(item.get('release_year'))}."
    )


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
    """Create a fingerprint of the prepared document."""
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# 4. SYNCHRONIZE THE CSV WITH CHROMADB
# --------------------------------------------------

def sync_dataset():
    """
    Add new titles, update changed titles, and remove
    titles that no longer exist in the CSV.

    Unchanged documents do not need new embeddings.
    """

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {CSV_PATH}\n"
            "Place the CSV beside main.py."
        )

    df = pd.read_csv(CSV_PATH)

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
        metadata = {
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
            f"Database is up to date: "
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
        f"Database ready: {collection.count()} titles."
    )


# --------------------------------------------------
# 5. SEARCH THE DATABASE
# --------------------------------------------------

def search_titles(query, top_k=5, media_type=None, era=None):
    """
    Embed the user's query and retrieve the closest
    matching titles from ChromaDB.

    Optional filters:
      media_type: "TV Show" or "Movie"
      era: "old" or "recent"
    """

    if not query.strip():
        print("Please enter a search query.")
        return

    if collection.count() == 0:
        print("The database is empty.")
        return

    # A new query normally requires one embedding request.
    query_embedding = create_embeddings(
        [query],
        task_type="RETRIEVAL_QUERY",
    )[0]

    # Build optional metadata filters.
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

    where = None

    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}

    # ChromaDB handles nearest-neighbor retrieval.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = results["ids"][0]

    if not ids:
        print("No matching titles found.")
        return

    print(f"\nSearch: {query}")
    print("-" * 60)

    for record_id, document, metadata, dist in zip(
        ids,
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        # With cosine distance, similarity is 1 - distance.
        score = 1 - dist

        print(
            f"\n[{score:.3f}] "
            f"{metadata.get('title', 'Unknown title')}"
        )

        print(
            f"Type: {metadata.get('type', 'Unknown')} | "
            f"Year: {metadata.get('release_year', 'Unknown')}"
        )

        print(f"Genres: {metadata.get('genres', 'Unknown')}")

        # The stored document includes the enriched fields.
        print(document)


# --------------------------------------------------
# 6. INTERACTIVE TERMINAL
# --------------------------------------------------

def main():
    # Only new or changed catalog entries are embedded.
    sync_dataset()

    print("\nTV & Movie Semantic Search")
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

        search_titles(
            query=query,
            top_k=top_k,
            media_type=media_type,
            era=era,
        )


if __name__ == "__main__":
    main()
