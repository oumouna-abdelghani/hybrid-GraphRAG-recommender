import os
import numpy as np
from dotenv import load_dotenv
from google import genai
from scipy.spatial import distance

# Load environment variables from .env file
load_dotenv()

# Initialize Gemini client (automatically reads GEMINI_API_KEY from .env)
client = genai.Client()

# Custom TV shows dataset
tv_shows = [
    {
        "id": 1,
        "title": "Better Call Saul",
        "description": "A shrewd, small-time defense attorney transforms into a morally flexible lawyer operating in the criminal underworld.",
        "category": "Crime Drama",
    },
    {
        "id": 2,
        "title": "Prison Break",
        "description": "A brilliant structural engineer purposefully gets sent to a high-security prison to break out his wrongfully convicted brother.",
        "category": "Action Thriller",
    },
    {
        "id": 3,
        "title": "Dark",
        "description": "A missing child setting off a frantic search leads four families to uncover a mind-bending time travel conspiracy spanning three generations.",
        "category": "Sci-Fi Mystery",
    },
    {
        "id": 4,
        "title": "From",
        "description": "Unwilling residents trapped in a nightmarish, inescapable town battle terrifying nocturnal creatures and mystery-filled woods.",
        "category": "Horror Mystery",
    },
    {
        "id": 5,
        "title": "Stranger Things",
        "description": "In the 1980s, a small Indiana town uncovers supernatural forces, secret government experiments, and an otherworldly dimension called the Upside Down.",
        "category": "Sci-Fi Horror",
    },
    {
        "id": 6,
        "title": "Suits",
        "description": "A brilliant college dropout with a photographic memory lands a high-stakes corporate legal job at a top Manhattan firm despite having no law degree.",
        "category": "Legal Drama",
    },
]


# Reusable embedding function for Gemini
def create_embeddings(texts):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=texts,
    )
    # Extract the vector (list of floats) for each input text
    return [e.values for e in response.embeddings]


# 1. Embed all TV show descriptions in a single batch call
print("Generating Gemini embeddings for your show watchlist...")
descriptions = [show["description"] for show in tv_shows]
show_embeddings = create_embeddings(descriptions)

# Store each embedding back in its show dictionary
for i, show in enumerate(tv_shows):
    show["embedding"] = show_embeddings[i]


# 2. Semantic search function
def search_shows(query, top_k=2):
    # Always pass a list into create_embeddings
    query_embedding = create_embeddings([query])[0]

    distances = []
    for show in tv_shows:
        dist = distance.cosine(query_embedding, show["embedding"])
        distances.append((dist, show))

    # Sort by lowest distance (closest match first)
    distances.sort(key=lambda item: item[0])

    print(f"\n🔍 Query: '{query}'")
    print("-" * 50)
    for dist, show in distances[:top_k]:
        score = 1 - dist  # cosine similarity
        print(f"[score {score:.3f}] {show['title']} ({show['category']})")
        print(f"   Summary: {show['description']}\n")


# 3. Test queries matching subtle themes
search_shows("lawyers dealing with high stakes corporate cases or illegal business")
search_shows("mind bending sci-fi puzzle involving time travel and hidden secrets")
search_shows("people trapped in a dangerous place trying to escape")