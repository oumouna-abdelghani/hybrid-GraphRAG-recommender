# Hybrid GraphRAG Movie & TV Recommendation Engine 🎬

A hybrid movie and TV recommendation engine combining **semantic vector search**, **knowledge graph traversal**, and **grounded LLM-generated explanations**.

The system uses ChromaDB to retrieve titles based on plot and thematic similarity, Neo4j to model relationships between titles and people, and Google Gemini to generate concise explanations grounded in retrieved information.

Rather than relying on an LLM to generate database queries, the engine uses deterministic entity extraction, parameterized Cypher queries, and Python-based ranking to combine semantic relevance with explicit relationships.

## Features

- **Hybrid Search Fusion:** Combines semantic similarity with graph-based bonuses to rank recommendations.

  `Final Score = Semantic Similarity Score + Graph Bonus`

- **Entity-Aware Filtering:** Recognizes actors and directors in natural-language queries and uses graph relationships to restrict eligible results.

- **Relationship-Based Ranking:** Boosts relevant titles connected to a seed title through shared cast members, directors, creators, producers, or franchise relationships.

- **Persistent Vector Search:** Stores embeddings in ChromaDB for reuse across sessions.

- **Incremental Vector Synchronization:** Uses SHA-256 fingerprints to identify changed records and avoid unnecessarily re-embedding unchanged titles.

- **Knowledge Graph Integration:** Represents titles, people, genres, and franchises as Neo4j nodes connected by typed relationships.

- **Grounded RAG Explanations:** Optionally uses Google Gemini to generate concise recommendation explanations based on retrieved plot descriptions and graph facts.

- **Metadata Filtering:** Supports filtering by media type (`Movie` or `TV Show`) and release era.

- **Interactive CLI:** Accepts natural-language queries and displays ranked recommendations, with optional generated explanations.

## Tech Stack

- **Language:** Python 3.10+
- **Vector Database:** ChromaDB
- **Knowledge Graph:** Neo4j
- **Embeddings:** Google Gemini API (`gemini-embedding-001`)
- **Text Generation:** Google Gemini API (`gemini-3.5-flash-lite`)
- **Data Processing:** Pandas
- **Other Tools:** NumPy, SciPy, regular expressions, Unicode normalization, SHA-256 hashing

## System Architecture

```text
                         User Query
                             |
                 Entity Extraction (Regex)
                             |
                +------------+------------+
                |                         |
          Gemini Embeddings        Neo4j Graph Search
                |                  Entity Constraints
          ChromaDB Search           Graph Connections
                |                         |
                +------------+------------+
                             |
                       Score Fusion
                             |
                  Ranked Recommendations
                             |
                   Retrieved Context
                             |
                    [Optional] Gemini
                    RAG Explanations
```

### How It Works

1. **Catalog Synchronization**

   - Loads movie and TV show metadata from a CSV file.
   - Builds enriched text representations for embedding.
   - Uses SHA-256 hashes to identify new, changed, and removed records in ChromaDB.
   - Builds the Neo4j graph with `Title`, `Person`, `Genre`, and `Franchise` nodes.

2. **Entity Recognition**

   - Uses an in-memory lookup index to recognize people and seed titles mentioned in natural-language queries.
   - Identifies when a user asks for titles similar to a particular movie or show.

3. **Graph Retrieval**

   - Executes parameterized Cypher queries to find titles connected through people or franchises.
   - Applies hard constraints when recognized entities must be matched.
   - Calculates graph-based bonuses for relevant connections to seed titles.

4. **Vector Search and Score Fusion**

   - Embeds the user's query and retrieves semantically similar candidates from ChromaDB.
   - Applies supported metadata filters.
   - Combines semantic relevance with graph-based bonuses to produce ranked results.

5. **Optional RAG Explanations**

   - Passes retrieved descriptions and graph facts to Gemini.
   - Generates short explanations for why each recommendation may fit the user's request.
   - Uses retrieved context to ground the explanations and reduce unsupported claims.

## Dataset

The engine loads movie and TV show metadata from `tv_movies_semantic_search_dataset.csv`.

The dataset includes title information such as names, plot descriptions, genres, themes, moods, and release years. Optional metadata about cast members, directors, creators, producers, and franchises enables graph-based retrieval through Neo4j.

List-valued fields use semicolons (`;`) as separators.

The `id` field should uniquely identify each title. The CSV columns and required fields must match the expectations of the data-loading code.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- A Google Gemini API key from [Google AI Studio](https://aistudio.google.com/)
- A running Neo4j database, such as a local instance managed by [Neo4j Desktop](https://neo4j.com/docs/desktop/current/)
- The project CSV dataset

### 1. Set Up Neo4j Locally

1. Install and open Neo4j Desktop.
2. Create a project and a local DBMS.
3. Set a database password and start the DBMS.
4. Check the connection details shown by Neo4j Desktop.

For a typical local setup, the connection settings are:

- URI: `bolt://localhost:7687`
- Username: `neo4j`
- Password: The password you configured

If Neo4j Desktop assigns a different Bolt port, use the URI shown in Desktop instead.

### 2. Clone the Repository

Replace the placeholder URL with your actual GitHub repository URL.

```bash
git clone https://github.com/YOUR-USERNAME/hybrid-graphrag-recommender.git
cd hybrid-graphrag-recommender
```

### 3. Install Dependencies

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```dotenv
# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite

# Neo4j Local Database
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
```

Replace the placeholder values with your actual credentials.

**Security:** Add `.env` to `.gitignore`. Never commit API keys or database passwords. You can provide a `.env.example` file containing placeholder values for other developers.

### 5. Run the Application

Make sure your Neo4j database is running and your CSV dataset is available at the expected path.

```bash
python main.py
```

Follow the terminal prompts to search for movies and TV shows, apply metadata filters, and optionally generate explanations.

## Project Evolution

This project evolved through three stages, progressing from basic embedding experiments to a persistent hybrid retrieval system.

### Phase 1 — In-Memory Semantic Search

Explored embedding generation and cosine similarity through small Python experiments, focusing on how text representations and query embeddings enable semantic retrieval.

### Phase 2 — Persistent Vector Search

Built a CSV-based semantic search backend with enriched embedding text, persistent ChromaDB storage, incremental synchronization, and interactive search with metadata filters.

**Representative commit:**

`feat: build persistent CSV-based semantic search backend`

### Phase 3 — Hybrid GraphRAG Retrieval

Integrated Neo4j to model relationships between titles and people, combined graph retrieval with vector search, and introduced optional Gemini-generated explanations grounded in retrieved context.

The implementation uses deterministic entity extraction and predefined Cypher queries rather than unconstrained LLM-generated database queries.

**Representative commit:**

`feat: implement hybrid GraphRAG retrieval with Neo4j and grounded Gemini explanations`

## Future Improvements

- [ ] **Streamlit Web UI:** Build an interactive interface for conversational queries, ranked recommendations, metadata filters, and explanations.
- [ ] **Recommendation Personalization:** Incorporate user feedback, liked titles, and viewing history to improve recommendations.
- [ ] **Evaluation:** Measure retrieval quality and compare semantic-only ranking against hybrid ranking.
- [ ] **MCP Integration:** Expose the recommendation pipeline through a Model Context Protocol (MCP) server for use by compatible AI applications.
- [ ] **Deployment:** Deploy the application with an appropriately hosted graph database and vector storage.

## note

This project started as a hands-on application of concepts explored in DataCamp coursework, including Introduction to Embeddings with the OpenAI API, Developing LLM Applications with LangChain, and Retrieval Augmented Generation (RAG) with LangChain.
