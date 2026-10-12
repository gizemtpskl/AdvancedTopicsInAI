# Advanced Topics in AI: Movie Chatbot (MaroonRollingStone)

A movie question-answering and recommendation chatbot built for the *Advanced Topics in AI* (ATAI) course at the University of Zurich. The bot runs on the course's **Speakeasy** chat platform and answers questions using a movie **knowledge graph**, **pre-trained graph embeddings**, and **MovieLens** data.

Everything lives in one notebook: [`ATAI_FINAL_MaroonRollingStone.ipynb`](ATAI_FINAL_MaroonRollingStone.ipynb).

## What the bot can do

| Capability | How it works |
|---|---|
| **Factual questions** (e.g. "Who directed *Fight Club*?") | Extracts the movie/person and the relation from the question, builds ranked SPARQL queries, and runs them on the knowledge graph. |
| **Embedding-based answers** | Looks up the entity and relation in the graph embeddings and returns the closest entities by cosine similarity. |
| **Movie recommendations** | Combines three signals from the movies a user likes: knowledge-graph metadata (genres, directors, actors, years), a fast random-walk over the graph, and MovieLens genre/plot-embedding similarity. |
| **Multimedia answers** | Returns an actor's profile picture, or a movie's poster or backdrop scene, using IMDb IDs and the course image index. |
| **Actor and movie lookups** | Handles questions like "What does the cast of *X* look like?" or "Show me a movie starring *Y*". |

The user can choose the answering strategy by phrasing the question, for example "... with a factual approach" or "... with an embedding approach". Recommendation and picture requests are detected automatically.

## How it works

1. **Load resources:** knowledge graph (RDF), entity/relation embeddings, label maps (entity to label and back).
2. **Understand the question:** entity extraction with a cascade of quoted text, BERT NER, spaCy, proper nouns and title-case fallback; fuzzy matching of titles to graph labels (RapidFuzz / TheFuzz).
3. **Find the relation:** hybrid scoring of keyword match, Sentence-BERT similarity and predicate labels to pick the most likely graph relation.
4. **Answer:** SPARQL query builder (top-k candidate queries), embedding lookup, recommendation pipeline, or multimedia lookup.
5. **Reply:** a Speakeasy `Agent` class listens for messages in the chat room and posts the answer.

## Tech stack

Python, Jupyter, `rdflib`, `sentence-transformers` (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`), `transformers` (`dslim/bert-base-NER`), `spaCy`, `NLTK`, `RapidFuzz` / `TheFuzz`, `scikit-learn`, `PyTorch`, `pandas`, `NumPy`, and `speakeasypy`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install rdflib sentence-transformers transformers torch spacy nltk \
            rapidfuzz thefuzz scikit-learn pandas numpy matplotlib speakeasypy
python -m spacy download en_core_web_sm
```

### Data (not included in this repository)

The datasets are too large for GitHub and are provided through the course. Place them as follows:

```
dataset/
├── graph.nt                      # movie knowledge graph
├── embeddings/
│   ├── entity_embeds.npy
│   ├── relation_embeds.npy
│   ├── entity_ids.del
│   └── relation_ids.del
└── additional/
    └── images.json               # IMDb ID to image index
```

The recommendation part also uses the [MovieLens](https://grouplens.org/datasets/movielens/) files (`movies.csv`, `ratings.csv`, `tags.csv`, `links.csv`) plus plot and comment data. The notebook generates its cached files (`combined_movie_embeddings_norm.npy`, `movielens_embeddings.pt`, `nn_index.pkl`, `movie_ids.json`) on the first run.

### Run the bot

1. Open the notebook and run the cells from top to bottom.
2. Set your own Speakeasy credentials for the bot (see the note below), then run the final cell. It starts the `Agent` and keeps listening for messages at `https://speakeasy.ifi.uzh.ch`.
3. Chat with the bot on the Speakeasy platform.

> **Credentials:** never commit a real username or password. Read them from environment variables instead, for example `os.environ["SPEAKEASY_USER"]` and `os.environ["SPEAKEASY_PASS"]`.

## Repository contents

- `ATAI_FINAL_MaroonRollingStone.ipynb`: the full chatbot implementation.
- `README.md`: this file.

## Course context

Developed as part of the *Advanced Topics in AI* course, University of Zurich (HS25).
