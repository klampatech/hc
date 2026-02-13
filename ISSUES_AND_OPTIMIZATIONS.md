# Issues & Optimizations - AI Job Search System

## Current Status (After Optimization Pass)

**Overall:** The system exceeds core project requirements. Token budget is well within targets,
and all queries return clearly relevant results. Several optimizations have been applied since
the initial implementation.

### Key Metrics

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| Total demo tokens | < $10 spend | 3,738 tokens ($0.0006) | PASS |
| Per search query | < 500 tokens avg | 403 tokens avg | PASS |
| Per refinement turn | 200-400 tokens | 219 tokens avg | PASS |
| Top 3 clearly relevant | All queries | 12/12 queries | PASS |
| Filters work correctly | All filter types | All working | PASS |
| No crashes/errors | Full demo | Clean run | PASS |

---

## Optimizations Applied

### 1. Result Deduplication (DONE)

**Problem:** Source dataset contains duplicate job postings (e.g., "Data Engineer - Ventura TRAVEL"
appeared twice in Flow 1 Turn 3).

**Fix:** Added `_deduplicate_results()` in search_engine.py that removes duplicates by
(title, company) key while preserving result ordering. Also fetches `top_k + 5` candidates
before dedup to compensate for removed duplicates.

### 2. Title Boost Stopwords (DONE)

**Problem:** Q5 "entry level design roles" was boosting civil engineering roles because
"entry" and "level" were being treated as meaningful title keywords.

**Fix:** Added `_TITLE_BOOST_STOPWORDS` frozenset that filters out generic terms.
Now Q5 correctly shows design roles (Associate Designer, Design Intern, etc.).

### 3. Seniority Title Fallback (DONE)

**Problem:** Many jobs lack structured seniority data, so the seniority filter was
ineffective for queries like "senior level only".

**Fix:** Enhanced seniority filter to also match title-based indicators:
- "senior" matches: senior, sr., sr, lead, principal, staff, head of, director, vp
- "entry" matches: junior, jr., intern, entry, associate
- "lead" matches: lead, principal, staff

### 4. Expanded Startup Detection (DONE)

**Problem:** Very few ML-specific roles had `org_type` tagged as "startup", causing
Flow 2 Turn 2 to show generic engineering roles instead of ML roles.

**Fix:** Expanded startup filter to include:
- Jobs with org_type containing "startup" (original)
- Companies with early-stage funding (seed, series a/b, pre-seed, angel)
- Companies with <= 50 employees

Combined with a +0.10 score bonus for jobs that have explicit "startup" in org_type,
so labeled startups rank higher while the expanded pool provides coverage for niche queries.

### 5. Data Loader: Funding Stage (DONE)

**Problem:** `funding_stage` field was always null because the data uses
`latest_investment_series` instead.

**Fix:** Updated data_loader.py to fall back to `latest_investment_series` for
funding stage data.

### 6. Lowered MIN_RESULTS Threshold (DONE)

**Problem:** The graceful filter degradation threshold of 10 was too high, causing
important filters (like seniority) to be silently dropped for niche combinations.

**Fix:** Lowered from 10 to 5. Better to show 5 highly relevant results than 10
loosely matching ones.

---

## Remaining Known Issues

### Issue 1: Limited Remote + Nonprofit Data Science Roles

**Severity:** Low (data limitation)

The intersection of remote + nonprofit + data science yields only ~8 results in the
dataset. Some results in Flow 1 Turn 3 are "Data Engineer" or "Data Analyst" rather
than "Data Scientist" - this is expected given the small pool.

### Issue 2: "Unknown" Company Names

**Severity:** Very Low (data quality)

A small number of results display "Unknown" as company name. These jobs lack company
info in all fallback locations. Not worth filtering out since the job data is otherwise
relevant.

---

## Validation Summary by Query

### Single-Turn Searches

| # | Query | Quality | Notes |
|---|-------|---------|-------|
| 1 | "senior software engineer remote" | Excellent | All 5 are senior + remote + SW engineer |
| 2 | "data science internships" | Excellent | #1-4 are actual DS/data interns |
| 3 | "product manager roles at early stage startups" | Excellent | Top 3 are PM roles at labeled startups |
| 4 | "backend engineer jobs in New York paying over 150k" | Excellent | All backend, >$150K, NY-area |
| 5 | "entry level design roles" | Good | Associates, interns, entry-level designers |
| 6 | "mission-driven nonprofit data roles" | Excellent | All data roles at nonprofits |

### Multi-Turn Refinements

| Flow | Turn | Query | Quality | Notes |
|------|------|-------|---------|-------|
| 1 | 1 | "data science jobs" | Excellent | All 5 are data science roles |
| 1 | 2 | "at companies...social good" | Excellent | All 5 are data science + nonprofit |
| 1 | 3 | "make it remote" | Good | Remote + nonprofit, no duplicates |
| 2 | 1 | "machine learning engineer" | Excellent | All 5 are ML engineer roles |
| 2 | 2 | "at early stage startups" | Excellent | All 5 are ML/AI roles at small companies |
| 2 | 3 | "senior level only" | Good | Senior ML roles (Senior, Principal, Staff) |

---

## Future Optimization Opportunities

### 1. Embedding Cache for Repeated Queries
- **Impact:** ~50 token savings per repeated query
- **Effort:** ~30 minutes
- Cache query embeddings to avoid re-embedding similar queries across refinement turns

### 2. BM25 Hybrid for Exact Keyword Matching
- **Impact:** Better results for specific role/skill queries
- **Effort:** ~2-3 hours
- Add TF-IDF/BM25 scoring alongside embeddings + filters for exact title matching
