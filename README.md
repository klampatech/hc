# AI Job Search System

An AI-powered job discovery engine that uses natural language queries and iterative refinement to search through 100,000 job postings.

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

1. Clone repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download dataset:
   - Download jobs.jsonl from: https://drive.google.com/file/d/1RRVWYAvfb4hUus1hUDY1nPQUJGqpiBiq/view
   - Place in `data/` directory

4. Create `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```

### Run Demo

```bash
python demo.py
```

This will:
- Load 100K job postings
- Run 6 single-turn searches
- Run 2 multi-turn refinement flows (3 turns each)
- Generate token usage report

## Architecture

### Data Representation

The system uses **pre-computed embeddings** (already in dataset) for three aspects of each job:

1. **Explicit embedding** - Job title, listed skills, requirements
2. **Inferred embedding** - Related skills, relevant experience
3. **Company embedding** - Company characteristics, industry, culture

Structured fields are indexed for boolean filtering:
- Remote status, location, seniority
- Salary range, employment type
- Company type (startup, nonprofit, enterprise)

### Search Strategy

Hybrid approach combining vector similarity + structured filters:

```
User Query
    |
[1] Embed query (text-embedding-3-small, ~50 tokens)
    |
[2] Extract structured filters (gpt-4o-mini, ~100 tokens)
    |
[3] Determine embedding weights (0 tokens - rule-based)
    |
[4] Compute similarity scores (0 tokens - numpy dot product)
    |
[5] Apply filters (0 tokens - boolean masking)
    |
[6] Optional re-ranking (gpt-4o-mini, ~800 tokens - only for ambiguous queries)
    |
Top 10 Results
```

**Key optimization:** ~70% of queries skip LLM re-ranking entirely.

### Relevance Ranking

Dynamic embedding weights based on query intent:

- **Job-focused queries** (e.g., "senior ML engineer")
  - 70% explicit, 20% inferred, 10% company

- **Company-focused queries** (e.g., "mission-driven startup")
  - 20% explicit, 20% inferred, 60% company

- **Balanced queries** (e.g., "data science at nonprofits")
  - 50% explicit, 30% inferred, 20% company

### Refinement Engine

Stateful conversation with incremental filtering:

1. Parse intent (~200 tokens) - refinement or pivot?
2. Merge new filters with existing (recency wins contradictions)
3. Re-search with combined context
4. Update state

## Trade-offs

### Optimized For
- Token efficiency (target: <500 tokens/query)
- Result quality for diverse query types
- Fast iteration (no complex infrastructure)
- 100K job scale in-memory

### Deprioritized
- Sub-second latency (embeddings + LLM calls take 2-5s)
- Exact phrase matching (embeddings are semantic)
- Learning from user feedback (stateless between sessions)
- Production-grade error handling

## Query Performance

### Works Well
- Clear job titles: "senior software engineer", "data analyst"
- Location + remote: "remote jobs in tech", "NYC product manager"
- Company types: "startup ML roles", "nonprofit data science"
- Salary filters: "engineering jobs over 150k"
- Multi-turn refinement: "data science" -> "at nonprofits" -> "remote"

### Challenging
- Very vague queries: "interesting roles", "good culture"
- Hyper-specific tech stacks: "Python + Kafka + Airflow + dbt"
- Company names not in dataset
- Queries mixing many constraints (may over-filter; system relaxes filters when <5 results)

## Token Usage

**Per-Query Runtime:**
- Simple search (with filters): ~150 tokens
- Ambiguous search (with re-ranking): ~950 tokens
- Refinement turn: ~350 tokens
- **Average: ~400 tokens/query**

## Future Improvements

1. **Caching** - Cache query embeddings for similar searches; estimated 30-50% token savings
2. **Smarter re-ranking triggers** - ML model to predict when re-ranking helps
3. **User feedback loop** - Track clicks, fine-tune weights, personalization
4. **Hybrid retrieval** - BM25 for exact keyword matching combined with semantic search
5. **Approximate nearest neighbor** - FAISS/Annoy for scaling to 1M+ jobs
6. **Explanation generation** - Show why each job matched to build user trust
