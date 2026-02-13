# AI Job Search System

An AI-powered job discovery engine that uses natural language queries and iterative refinement to search through 100,000 job postings.

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

1. Clone repository
2. Create virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
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

This runs 6 single-turn searches and 2 multi-turn refinement flows (3 turns each) against 100K job postings, then generates a token usage report.

### Interactive Mode

```bash
python interactive.py
```

A REPL-style search bar where you can type queries and refine results conversationally. Commands: `/new` to reset, `/tokens` to see usage, `/quit` to exit.

## Architecture

### Project Structure

```
src/
  data_loader.py    - Loads jobs.jsonl, extracts structured fields, builds embedding matrices
  search_engine.py  - Hybrid search: embeddings + filters + optional LLM re-ranking
  refinement_engine.py - Multi-turn state management with rule-based and LLM intent parsing
  utils.py          - Vector normalization, safe nested dict access
demo.py             - Automated demo with 12 queries and token report
interactive.py      - Interactive search REPL
```

### Data Representation

The system uses **pre-computed embeddings** (already in dataset) for three aspects of each job:

1. **Explicit embedding** - Job title, listed skills, requirements
2. **Inferred embedding** - Related skills, relevant experience
3. **Company embedding** - Company characteristics, industry, culture

All embeddings are 1536-dimensional vectors from OpenAI `text-embedding-3-small`, pre-normalized into NumPy matrices at load time for fast batch dot-product similarity.

Structured fields are indexed for boolean filtering:
- Remote status, location, seniority
- Salary range, employment type
- Company type (startup, nonprofit, enterprise)
- Funding stage, employee count

### Search Pipeline

Hybrid approach combining vector similarity + structured filters:

```
User Query
    |
[1] Embed query (text-embedding-3-small, ~5 tokens)
    |
[2] Extract structured filters (gpt-4o-mini, ~150 tokens)
    |
[3] Determine embedding weights (0 tokens - rule-based heuristic)
    |
[4] Compute weighted similarity scores (0 tokens - numpy dot product)
    |
[4b] Title keyword boost (0 tokens - string matching)
    |
[5] Apply filters progressively (0 tokens - boolean masking with graceful degradation)
    |
[6] Deduplicate results by (title, company)
    |
[7] Optional re-ranking (gpt-4o-mini, ~800 tokens - only for ambiguous queries)
    |
Top 10 Results
```

**Key optimization:** Most queries skip LLM re-ranking entirely. Only queries with ambiguous/subjective keywords (e.g., "culture", "mission-driven", "innovative") trigger re-ranking.

### Relevance Ranking

Dynamic embedding weights based on query intent (rule-based, 0 tokens):

- **Job-focused queries** (e.g., "senior ML engineer")
  - 70% explicit, 20% inferred, 10% company

- **Company-focused queries** (e.g., "mission-driven startup")
  - 20% explicit, 20% inferred, 60% company

- **Balanced queries** (e.g., "data science at nonprofits")
  - 50% explicit, 30% inferred, 20% company

A **title keyword boost** adds a small score bonus (+0.05 per match) for jobs whose title contains query keywords, helping maintain role focus when filters narrow the candidate pool. A stopword list prevents generic terms like "entry", "level", "roles" from inflating irrelevant matches.

### Filter System

Filters are applied **progressively**: each filter is only kept if it doesn't reduce the candidate pool below 5 results. This prevents over-filtering on niche query combinations while preserving as many constraints as possible.

Special handling:
- **Startup detection** expands beyond the `org_type` label to include companies with early-stage funding (seed, series A/B, pre-seed, angel) or <= 50 employees, with a +0.10 score bonus for explicitly labeled startups
- **Seniority fallback** matches title-based indicators (e.g., "Sr.", "Lead", "Principal", "Staff") when structured seniority data is missing
- **Remote + location contradiction** automatically drops the location filter when remote is specified

### Refinement Engine

Multi-turn search with stateful conversation tracking:

1. **Rule-based fast path (0 tokens)** - Short filter-only inputs like "make it remote", "at startups", "senior level only" are parsed with pattern matching and applied instantly
2. **LLM intent parsing (~150 tokens)** - For complex follow-ups, GPT-4o-mini classifies the intent (refinement, pivot, or clarification) and extracts new filters
3. **Filter merging** - New filters are merged with existing ones (new values override old; remote overrides location)
4. **Re-search** - The original query embedding is reused with updated filters, preserving role focus

## Demo Results

Results from the latest demo run (12 queries total):

### Single-Turn Searches

| # | Query | Top Result | Quality |
|---|-------|-----------|---------|
| 1 | "senior software engineer remote" | Senior Full Stack SW Engineer - Zealogics (Remote) | All 5 senior + remote + SW engineer |
| 2 | "data science internships" | Data Science Intern - XPO, Inc. (Boston) | Top 4 are actual DS/data interns |
| 3 | "product manager roles at early stage startups" | Senior PM - Jerry.ai (Remote, Tech Startup) | Top 3 are PM roles at labeled startups |
| 4 | "backend engineer jobs in New York paying over 150k" | Backend SW Engineer - Medal ($180K-$250K, NYC) | All backend, >$150K, NY-area |
| 5 | "entry level design roles" | Design Professional - McMillan Pazdan Smith | Associates, interns, entry-level designers |
| 6 | "mission-driven nonprofit data roles" | Donor Data Specialist - Springs Rescue Mission | All data roles at nonprofits |

### Multi-Turn Refinement Flows

**Flow 1: Data Science -> Social Good -> Remote**

| Turn | Query | Active Filters | Top Result |
|------|-------|---------------|------------|
| 1 | "data science jobs" | none | Senior Manager, Data Science - American Express |
| 2 | "at companies or non-profits that care about social good" | nonprofit | Senior Data Science Analyst - Mayo Clinic |
| 3 | "make it remote" | remote, nonprofit | Senior Data Science Analyst - Mayo Clinic (Remote) |

**Flow 2: ML Engineer -> Startups -> Senior**

| Turn | Query | Active Filters | Top Result |
|------|-------|---------------|------------|
| 1 | "machine learning engineer" | none | ML Engineer - CompassMSP ($120K-$200K) |
| 2 | "at early stage startups" | startup | Product Engineer - Advocate (Remote, Tech Startup) |
| 3 | "senior level only" | senior, startup | Senior SW Engineer - Overland AI ($200K-$240K, Startup) |

## Trade-offs

### Optimized For
- **Token efficiency** - Average 402 tokens/search, 259 tokens/refinement
- **Result quality** - 12/12 demo queries return clearly relevant top results
- **Fast iteration** - No database, no indexing server, pure in-memory NumPy
- **100K scale** - All jobs loaded and searchable in ~60 seconds

### Deprioritized
- Sub-second latency (embedding + LLM calls take 2-5s per query)
- Exact phrase matching (embeddings are semantic, not lexical)
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
- Very vague queries: "interesting roles", "good culture" (triggers re-ranking to help)
- Hyper-specific tech stacks: "Python + Kafka + Airflow + dbt"
- Company names not in dataset
- Queries mixing many constraints (graceful degradation drops lowest-impact filters when <5 results)

## Token Usage

**Demo totals (12 queries):**
- Single-turn searches (6 queries): 2,415 tokens
- Multi-turn refinements (6 turns): 1,556 tokens
- **Total: 3,971 tokens (~$0.0006 with GPT-4o-mini)**

**Per-query averages:**
- Single-turn search: ~402 tokens
- Refinement turn: ~259 tokens

**Breakdown by operation:**
| Operation | Tokens | Purpose |
|-----------|--------|---------|
| Filter extraction | 1,798 | GPT-4o-mini extracts structured filters from queries |
| Intent parsing | 1,104 | GPT-4o-mini classifies refinement intent |
| Re-ranking | 1,015 | GPT-4o-mini re-ranks ambiguous queries (1 of 6 searches) |
| Embedding | 54 | text-embedding-3-small for query vectors |

## Future Improvements

1. **Embedding cache** - Cache query embeddings for similar/repeated searches; ~50 token savings per repeated query
2. **BM25 hybrid retrieval** - Add TF-IDF/BM25 scoring alongside embeddings for exact keyword matching
3. **Smarter re-ranking triggers** - ML model to predict when re-ranking helps vs. wastes tokens
4. **User feedback loop** - Track clicks, fine-tune weights, personalization
5. **Approximate nearest neighbor** - FAISS/Annoy for scaling to 1M+ jobs
6. **Explanation generation** - Show why each job matched to build user trust
