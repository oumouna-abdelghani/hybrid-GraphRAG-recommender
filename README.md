# Semantic Search Engine for Movies & TV Shows 🎬

A Python-based semantic search engine that helps users discover movies and TV shows using natural-language queries. Built with Google's Gemini Embedding API and ChromaDB, the project explores enriched text embeddings, vector similarity search, persistent storage, and metadata filtering.

## Features

- **Enriched Text Embeddings:** Combines titles, descriptions, genres, themes, moods, and release years into richer text representations.
- **CSV Dataset:** Loads movie and TV-show information from a structured CSV file, making the catalog easy to expand and maintain.
- **Semantic Search:** Retrieves relevant titles based on the meaning of a user's natural-language query rather than relying only on exact keyword matches.
- **Persistent Vector Database:** Stores embeddings in ChromaDB so catalog vectors remain available between program runs.
- **Incremental Updates:** Detects new, modified, and deleted catalog entries to avoid unnecessarily regenerating embeddings for unchanged titles.
- **Top-K Retrieval:** Returns the most relevant matches ranked by cosine similarity.
- **Interactive Terminal Search:** Allows users to enter their own queries and choose the number of results.
- **Metadata Filtering:** Supports filtering by media type and release period.

## Tech Stack

- Python
- Google Gemini API (`google-genai`)
- ChromaDB
- Pandas
- python-dotenv

## How It Works

1. Load movie and TV-show data from a CSV dataset.
2. Combine relevant metadata into enriched text representations using Python f-strings.
3. Generate embeddings for new or modified entries using Google's Gemini Embedding API.
4. Store the embeddings, documents, and metadata in a persistent ChromaDB collection.
5. Convert the user's query into an embedding.
6. Retrieve the nearest vectors from ChromaDB and display the highest-ranked results.
7. Synchronize the database with the CSV when the catalog changes.

## Getting Started

### Prerequisites

- Python 3.10 or later
- A Google Gemini API key from Google AI Studio

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mini-semantic-search.git
cd mini-semantic-search
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure your API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Never commit your `.env` file or expose your API key publicly.

### 4. Prepare the dataset


The project includes a CSV dataset containing movies and TV shows, with metadata such as titles, descriptions, genres, themes, moods, and release years.

The application loads the dataset from tv_movies_semantic_search_dataset.csv and uses this information to generate enriched embeddings for semantic search.

To use a custom dataset, provide a CSV file with the required columns:

id
title
type
description
genres
themes
mood
release_year

Each entry must have a unique ID.
### 5. Run the project

```bash
python main.py
```

Enter a natural-language query, choose optional filters, and explore the returned results.

## Embedding and Storage

The project uses Google's `gemini-embedding-001` model to generate document and query embeddings.

ChromaDB persists the catalog vectors locally. On subsequent runs, the application compares document fingerprints to identify changes and only generates new catalog embeddings when necessary.

**Note:** Catalog embeddings are reused when documents remain unchanged, but new search queries generally require their own embedding API request.

The local ChromaDB directory is excluded from version control. When cloning the repository, the database is created and populated automatically from the CSV dataset.

## Future Improvements

- Add an interactive web interface using Streamlit.
- Introduce personalized recommendations based on user preferences and ratings.
- Improve retrieval quality through experiments with text representations and ranking strategies.
- Evaluate search results using a collection of representative queries.
- Explore additional embedding models and vector database configurations.

## Learning Goals
I made this project to wrap up the data camp course " introduction to embeddings with the openAI API "
This project is a practical exploration of text embeddings, semantic retrieval, vector databases, metadata filtering, and efficient data synchronization. It is designed to evolve incrementally as new concepts and capabilities are introduced.
