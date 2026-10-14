# Semantic Search Engine for Movies & TV Shows 🎬

A small Python project exploring semantic search using text embeddings and the Google Gemini API.

Built as a hands-on practice project exploring vector embeddings, originally inspired by DataCamp's embeddings concepts and implemented using Google's Gemini SDK.

## Features

- **Text Embeddings:** Converts movie and TV show descriptions into numerical vectors using Google's `text-embedding-004` model.
- **Batch Embedding:** Generates embeddings for multiple descriptions in a single API request.
- **Semantic Search:** Finds relevant titles based on the meaning of a natural-language query.
- **Cosine Distance:** Measures the similarity between query and catalog embeddings.
- **Top-K Retrieval:** Sorts results and returns the closest matches.

## Tech Stack

- Python
- Google Gemini API (`google-genai`)
- SciPy
- NumPy

## How It Works

1. Store movie and TV show titles and descriptions in a dataset.
2. Generate embeddings for the descriptions using the Google Gemini API.
3. Convert a search query into an embedding.
4. Calculate cosine distance between the query and each catalog entry.
5. Sort the results and return the most relevant titles.

## Getting Started

### Prerequisites

- Python 3.10 or later recommended
- A Google Gemini API key (from Google AI Studio)

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mini-semantic-search.git
cd mini-semantic-search
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Configure your API key

Create a file named `.env` in the root directory and add your Gemini API key:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the project

```bash
python main.py
```

## Future Improvements

- Expand the catalog with more movies and TV shows.
- Load data from a CSV file.
- Save embeddings to avoid regenerating them on every run.
- Explore vector databases for larger datasets.
- Build an interactive interface using Streamlit.

## Notes

This is a small educational project focused on the fundamentals of embedding-based semantic search.
