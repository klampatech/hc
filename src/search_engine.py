import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI

from .data_loader import Job, JobDataset


@dataclass
class SearchFilters:
    """Structured filters extracted from query."""
    is_remote: Optional[bool] = None
    seniority: Optional[str] = None
    location: Optional[str] = None
    min_salary: Optional[float] = None
    org_type: Optional[str] = None
    industry: Optional[str] = None
    employment_type: Optional[str] = None


@dataclass
class SearchResult:
    """Search result with score."""
    job: Job
    score: float
    match_type: str = "hybrid"


FILTER_SYSTEM_PROMPT = """Extract structured filters from job search query.
Return ONLY a JSON object with these optional fields:
{
  "is_remote": true/false/null,
  "seniority": "entry"/"mid"/"senior"/"lead"/"director"/null,
  "location": "city, state"/null,
  "min_salary": number/null,
  "org_type": "startup"/"nonprofit"/null,
  "industry": "string"/null,
  "employment_type": "Full Time"/"Part Time"/"Contract"/null
}

Examples:
"remote senior engineer" -> {"is_remote": true, "seniority": "senior"}
"data scientist in NYC over 150k" -> {"location": "New York", "min_salary": 150000}
"nonprofit data roles" -> {"org_type": "nonprofit"}
"PM at startups" -> {"org_type": "startup"}
"entry level design" -> {"seniority": "entry"}"""


class SearchEngine:
    """Hybrid search using embeddings + structured filters."""

    def __init__(self, dataset: JobDataset, openai_api_key: str):
        self.dataset = dataset
        self.client = OpenAI(api_key=openai_api_key)
        self.token_usage = {"total": 0, "embedding": 0, "filter_extraction": 0, "reranking": 0, "intent_parsing": 0}
        self._last_extracted_filters = SearchFilters()

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters_override: Optional[SearchFilters] = None,
        use_reranking: bool = False,
    ) -> Tuple[List[SearchResult], int]:
        """Main search function. Returns (results, tokens_used)."""
        # Step 1: Embed the query
        query_embedding, tokens_embed = self._embed_query(query)

        # Step 2: Extract or use provided filters
        if filters_override is not None:
            filters = filters_override
            tokens_filter = 0
        else:
            filters, tokens_filter = self._extract_filters(query)

        self._last_extracted_filters = filters

        # Step 3: Determine embedding weights (rule-based, 0 tokens)
        weights = self._determine_embedding_weights(query, filters)

        # Step 4: Compute similarity scores (pure math, 0 tokens)
        scores = self._compute_similarity_scores(query_embedding, weights)

        # Step 4b: Title keyword boost (0 tokens, helps role focus in filtered pools)
        scores += self._title_match_boost(query)

        # Step 4c: Bonus for jobs that tightly match filter criteria
        if filters.org_type and filters.org_type.lower() == "startup":
            scores += np.array([
                0.10 if j.org_type and "startup" in j.org_type.lower() else 0.0
                for j in self.dataset.jobs
            ])

        # Step 5: Apply structured filters
        mask = self._apply_filters(filters)

        # Step 6: Get top candidates
        candidate_k = top_k * 3 if (use_reranking or self._should_rerank(query, filters)) else top_k + 5
        top_indices = self._get_top_k_indices(scores, mask, candidate_k)

        # Step 7: Optional re-ranking
        tokens_rerank = 0
        if use_reranking or self._should_rerank(query, filters):
            candidates = [self.dataset.jobs[i] for i in top_indices]
            reranked, tokens_rerank = self._rerank_with_llm(query, candidates[:30], top_k)
            results = [
                SearchResult(job=job, score=1.0 - rank * 0.01)
                for rank, job in enumerate(reranked)
            ]
        else:
            results = [
                SearchResult(job=self.dataset.jobs[i], score=float(scores[i]))
                for i in top_indices[:top_k]
            ]

        # Deduplicate by (title, company) preserving order
        results = self._deduplicate_results(results, top_k)

        total_tokens = tokens_embed + tokens_filter + tokens_rerank
        self.token_usage["total"] += total_tokens
        self.token_usage["embedding"] += tokens_embed
        self.token_usage["filter_extraction"] += tokens_filter
        self.token_usage["reranking"] += tokens_rerank

        return results, total_tokens

    def _embed_query(self, query: str) -> Tuple[np.ndarray, int]:
        """Embed query using text-embedding-3-small."""
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query,
        )
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        return embedding, response.usage.total_tokens

    def _extract_filters(self, query: str) -> Tuple[SearchFilters, int]:
        """Extract structured filters from query using GPT-4o-mini."""
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": FILTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=150,
        )

        try:
            content = response.choices[0].message.content.strip()
            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            filter_json = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            filter_json = {}

        # Build SearchFilters from parsed JSON, ignoring null values
        kwargs = {k: v for k, v in filter_json.items() if v is not None and hasattr(SearchFilters, k)}
        filters = SearchFilters(**kwargs)

        return filters, response.usage.total_tokens

    def _determine_embedding_weights(self, query: str, filters: SearchFilters) -> Dict[str, float]:
        """Determine weights for 3 embedding types based on query intent (rule-based, 0 tokens)."""
        query_lower = query.lower()

        company_keywords = [
            "startup", "culture", "mission", "nonprofit", "social good",
            "fast-growing", "early stage", "series", "funded",
        ]
        explicit_keywords = [
            "engineer", "developer", "scientist", "analyst", "manager",
            "designer", "senior", "junior", "lead", "product", "intern",
            "python", "react", "backend", "frontend", "fullstack", "data",
        ]

        company_score = sum(1 for kw in company_keywords if kw in query_lower)
        explicit_score = sum(1 for kw in explicit_keywords if kw in query_lower)

        # Only go company-heavy when company intent strongly dominates
        if company_score > explicit_score * 2:
            return {"explicit": 0.2, "inferred": 0.2, "company": 0.6}
        elif explicit_score > company_score * 2:
            return {"explicit": 0.7, "inferred": 0.2, "company": 0.1}
        else:
            return {"explicit": 0.5, "inferred": 0.3, "company": 0.2}

    def _compute_similarity_scores(self, query_embedding: np.ndarray, weights: Dict[str, float]) -> np.ndarray:
        """Compute weighted cosine similarity across all 3 embedding types."""
        query_norm = query_embedding / np.linalg.norm(query_embedding)

        # Embeddings are pre-normalized in dataset
        sim_explicit = self.dataset.embeddings_explicit @ query_norm
        sim_inferred = self.dataset.embeddings_inferred @ query_norm
        sim_company = self.dataset.embeddings_company @ query_norm

        return (
            weights["explicit"] * sim_explicit
            + weights["inferred"] * sim_inferred
            + weights["company"] * sim_company
        )

    # Generic terms that shouldn't contribute to title matching
    _TITLE_BOOST_STOPWORDS = frozenset({
        "entry", "level", "roles", "jobs", "stage", "early", "paying",
        "over", "remote", "that", "care", "about", "good", "make",
        "only", "companies", "with", "from", "looking",
    })

    def _title_match_boost(self, query: str) -> np.ndarray:
        """Small score boost for jobs whose title contains query keywords.
        Costs 0 tokens (pure string matching). Helps maintain role focus
        when filters restrict the candidate pool."""
        terms = [
            t.lower() for t in query.split()
            if len(t) > 3 and t.lower() not in self._TITLE_BOOST_STOPWORDS
        ]
        if not terms:
            return np.zeros(len(self.dataset))
        boost = np.zeros(len(self.dataset))
        for i, job in enumerate(self.dataset.jobs):
            tl = job.title.lower()
            boost[i] = sum(0.05 for t in terms if t in tl)
        return boost

    def _apply_filters(self, filters: SearchFilters) -> np.ndarray:
        """Apply structured filters progressively. Skips any filter that would
        reduce results below the threshold, so we keep as many filters as
        possible without over-restricting."""
        MIN_RESULTS = 5
        n = len(self.dataset)
        mask = np.ones(n, dtype=bool)
        jobs = self.dataset.jobs

        # Build individual filter masks
        individual_masks = []

        if filters.is_remote is not None:
            individual_masks.append(
                np.array([j.is_remote == filters.is_remote for j in jobs])
            )

        if filters.seniority:
            sl = filters.seniority.lower()
            # Match structured seniority OR title-based seniority (fallback
            # for the many jobs that lack structured seniority data)
            _SENIORITY_TITLE_MAP = {
                "senior": ("senior", "sr.", "sr ", "lead", "principal",
                           "staff", "head of", "director", "vp "),
                "lead": ("lead", "principal", "staff"),
                "entry": ("junior", "jr.", "jr ", "intern", "entry", "associate"),
                "mid": (),
                "director": ("director", "vp ", "vice president"),
            }
            title_hints = _SENIORITY_TITLE_MAP.get(sl, ())
            individual_masks.append(np.array([
                (j.seniority is not None and sl in j.seniority.lower())
                or any(h in j.title.lower() for h in title_hints)
                for j in jobs
            ]))

        if filters.location:
            ll = filters.location.lower()
            individual_masks.append(np.array([
                j.location is not None and ll in j.location.lower()
                for j in jobs
            ]))

        if filters.min_salary:
            individual_masks.append(np.array([
                j.salary_max is not None and j.salary_max >= filters.min_salary
                for j in jobs
            ]))

        if filters.org_type:
            ol = filters.org_type.lower()
            if ol == "startup":
                # Expanded startup detection: org_type label OR early-stage
                # funding OR very small company (<=50 employees)
                _STARTUP_FUNDING = {"seed", "series a", "series b",
                                    "pre-seed", "angel", "pre-ipo"}

                def _is_startup(j):
                    if j.org_type is not None and ol in j.org_type.lower():
                        return True
                    if (j.funding_stage is not None
                            and j.funding_stage.lower()
                            in _STARTUP_FUNDING):
                        return True
                    try:
                        if j.employee_count and int(j.employee_count) <= 50:
                            return True
                    except (ValueError, TypeError):
                        pass
                    return False

                individual_masks.append(
                    np.array([_is_startup(j) for j in jobs])
                )
            else:
                individual_masks.append(np.array([
                    j.org_type is not None and ol in j.org_type.lower()
                    for j in jobs
                ]))

        if filters.industry:
            il = filters.industry.lower()
            individual_masks.append(np.array([
                j.industry is not None and il in j.industry.lower()
                for j in jobs
            ]))

        if filters.employment_type:
            el = filters.employment_type.lower()
            individual_masks.append(np.array([
                j.employment_type is not None and el in j.employment_type.lower()
                for j in jobs
            ]))

        # Apply filters greedily: keep each filter only if the combined
        # result still has enough matches
        for m in individual_masks:
            candidate = mask & m
            if candidate.sum() >= MIN_RESULTS:
                mask = candidate

        return mask

    def _get_top_k_indices(self, scores: np.ndarray, mask: np.ndarray, k: int) -> np.ndarray:
        """Get indices of top k jobs after applying mask."""
        scores_filtered = scores.copy()
        scores_filtered[~mask] = -np.inf
        return np.argsort(scores_filtered)[::-1][:k]

    @staticmethod
    def _deduplicate_results(results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """Remove duplicate jobs by (title, company), preserving order."""
        seen = set()
        deduped = []
        for r in results:
            key = (r.job.title.lower().strip(), r.job.company.lower().strip())
            if key not in seen:
                seen.add(key)
                deduped.append(r)
                if len(deduped) >= top_k:
                    break
        return deduped

    def _should_rerank(self, query: str, filters: SearchFilters) -> bool:
        """Decide if query needs LLM re-ranking. Only for genuinely ambiguous queries."""
        query_lower = query.lower()

        ambiguous_keywords = [
            "culture", "mission", "innovative", "creative", "exciting",
            "growth", "opportunity", "interesting", "meaningful", "impact",
        ]

        if any(kw in query_lower for kw in ambiguous_keywords):
            return True

        return False

    def _rerank_with_llm(self, query: str, candidates: List[Job], top_k: int) -> Tuple[List[Job], int]:
        """Use GPT-4o-mini to re-rank top candidates."""
        job_summaries = []
        for i, job in enumerate(candidates):
            summary = f"{i}. {job.title} at {job.company}"
            if job.org_type:
                summary += f" ({job.org_type})"
            if job.is_remote:
                summary += " - Remote"
            if job.skills:
                summary += f" | {', '.join(job.skills[:5])}"
            job_summaries.append(summary)

        n_return = min(top_k, len(candidates))
        prompt = (
            f'Rank these jobs by relevance to: "{query}"\n\n'
            f"Jobs:\n{chr(10).join(job_summaries)}\n\n"
            f"Return ONLY a JSON array of {n_return} indices in ranked order:\n"
            f"[0, 5, 2, ...]"
        )

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )

        try:
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            ranked_indices = json.loads(content)
            valid_indices = [i for i in ranked_indices if isinstance(i, int) and 0 <= i < len(candidates)]
            reranked = [candidates[i] for i in valid_indices[:top_k]]
        except (json.JSONDecodeError, IndexError, TypeError):
            reranked = candidates[:top_k]

        return reranked, response.usage.total_tokens

    def track_external_tokens(self, operation: str, tokens: int):
        """Track tokens from external callers (e.g. refinement engine)."""
        self.token_usage["total"] += tokens
        if operation not in self.token_usage:
            self.token_usage[operation] = 0
        self.token_usage[operation] += tokens

    def get_token_report(self) -> str:
        """Generate token usage report."""
        report = f"Total tokens used: {self.token_usage['total']:,}\n\n"
        report += "By operation:\n"
        for op, tokens in self.token_usage.items():
            if op != "total" and tokens > 0:
                report += f"  {op}: {tokens:,} tokens\n"
        return report
