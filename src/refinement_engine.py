import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .search_engine import SearchEngine, SearchFilters, SearchResult


@dataclass
class ConversationState:
    """Track conversation state across turns."""
    active_filters: SearchFilters = field(default_factory=SearchFilters)
    last_query: str = ""
    last_results: List[SearchResult] = field(default_factory=list)
    conversation_history: List[str] = field(default_factory=list)


INTENT_PROMPT_TEMPLATE = """{context}New user input: "{new_query}"

Classify the intent and extract any new filters.

IMPORTANT: If the user is adding constraints to their existing search (location, remote,
seniority, company type, salary, industry), classify as "refinement" even if the input is
short. Only use "pivot" when the user clearly wants a DIFFERENT job role/title (e.g.
"show me marketing roles instead").

Return JSON:
{{
  "type": "refinement" | "pivot" | "show_different" | "clarification",
  "filters": {{
    "is_remote": true/false/null,
    "seniority": "entry"/"mid"/"senior"/"lead"/null,
    "location": "string"/null,
    "min_salary": number/null,
    "org_type": "startup"/"nonprofit"/"enterprise"/null,
    "industry": "string"/null
  }}
}}

Examples:
"make it remote" -> {{"type": "refinement", "filters": {{"is_remote": true}}}}
"remote" -> {{"type": "refinement", "filters": {{"is_remote": true}}}}
"at startups" -> {{"type": "refinement", "filters": {{"org_type": "startup"}}}}
"in New York" -> {{"type": "refinement", "filters": {{"location": "New York"}}}}
"over 200k" -> {{"type": "refinement", "filters": {{"min_salary": 200000}}}}
"actually show me marketing roles instead" -> {{"type": "pivot", "filters": {{}}}}
"at nonprofits" -> {{"type": "refinement", "filters": {{"org_type": "nonprofit"}}}}"""


class RefinementEngine:
    """Handle multi-turn search refinement."""

    # Multi-word filter phrases (checked before single words)
    _MULTI_WORD_FILTERS = [
        ("entry level", {"seniority": "entry"}),
        ("senior level", {"seniority": "senior"}),
        ("mid level", {"seniority": "mid"}),
        ("full time", {"employment_type": "Full Time"}),
        ("full-time", {"employment_type": "Full Time"}),
        ("part time", {"employment_type": "Part Time"}),
        ("part-time", {"employment_type": "Part Time"}),
        ("non profit", {"org_type": "nonprofit"}),
        ("non-profit", {"org_type": "nonprofit"}),
        ("non profits", {"org_type": "nonprofit"}),
        ("non-profits", {"org_type": "nonprofit"}),
        ("on site", {"is_remote": False}),
        ("on-site", {"is_remote": False}),
        ("in office", {"is_remote": False}),
        ("in-office", {"is_remote": False}),
    ]

    # Single-word filter keywords
    _SINGLE_WORD_FILTERS = {
        "remote": {"is_remote": True},
        "onsite": {"is_remote": False},
        "hybrid": {},
        "senior": {"seniority": "senior"},
        "junior": {"seniority": "entry"},
        "lead": {"seniority": "lead"},
        "director": {"seniority": "director"},
        "startup": {"org_type": "startup"},
        "startups": {"org_type": "startup"},
        "nonprofit": {"org_type": "nonprofit"},
        "nonprofits": {"org_type": "nonprofit"},
        "contract": {"employment_type": "Contract"},
    }

    # Words that carry no filter meaning on their own
    _FILLER_WORDS = frozenset({
        "make", "it", "only", "just", "at", "in", "for", "and", "with",
        "the", "a", "an", "please", "show", "me", "find", "get", "ones",
        "that", "are", "is", "should", "be", "to", "i", "want", "level",
        "or", "of", "type", "based", "roles", "jobs", "positions",
    })

    _SALARY_PATTERN = re.compile(
        r"(?:over|above|at\s+least|minimum|min|paying|>\s*)\s*\$?\s*(\d+)\s*k",
        re.IGNORECASE,
    )

    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
        self.client = search_engine.client

    def refine(
        self,
        state: ConversationState,
        new_query: str,
        top_k: int = 10,
    ) -> Tuple[List[SearchResult], ConversationState, int]:
        """Refine search based on conversation context. Returns (results, updated_state, tokens_used)."""
        # First query is always a fresh search
        if not state.last_query:
            results, tokens = self.search_engine.search(new_query, top_k)
            new_state = ConversationState(
                active_filters=self.search_engine._last_extracted_filters,
                last_query=new_query,
                last_results=results,
                conversation_history=[new_query],
            )
            return results, new_state, tokens

        # Try rule-based filter detection first (0 tokens, instant)
        rule_filters = self._try_rule_based_filters(new_query)
        if rule_filters is not None:
            merged_filters = self._merge_filters(state.active_filters, rule_filters)
            results, tokens_search = self.search_engine.search(
                state.last_query, top_k, filters_override=merged_filters,
            )
            new_state = ConversationState(
                active_filters=merged_filters,
                last_query=state.last_query,
                last_results=results,
                conversation_history=state.conversation_history + [new_query],
            )
            return results, new_state, tokens_search

        # Fall back to LLM intent parsing
        intent, tokens_intent = self._parse_refinement_intent(state, new_query)
        self.search_engine.track_external_tokens("intent_parsing", tokens_intent)

        if intent["type"] == "pivot":
            results, tokens_search = self.search_engine.search(new_query, top_k)
            new_state = ConversationState(
                active_filters=self.search_engine._last_extracted_filters,
                last_query=new_query,
                last_results=results,
                conversation_history=state.conversation_history + [new_query],
            )
            return results, new_state, tokens_intent + tokens_search

        elif intent["type"] == "refinement":
            new_filter_data = intent.get("filters", {})
            merged_filters = self._merge_filters(state.active_filters, new_filter_data)

            # Use original query for embedding (keeps role focus intact);
            # the new intent is captured by the merged filters.
            results, tokens_search = self.search_engine.search(
                state.last_query,
                top_k,
                filters_override=merged_filters,
            )

            new_state = ConversationState(
                active_filters=merged_filters,
                last_query=state.last_query,
                last_results=results,
                conversation_history=state.conversation_history + [new_query],
            )
            return results, new_state, tokens_intent + tokens_search

        else:  # show_different or clarification
            return state.last_results, state, tokens_intent

    def _try_rule_based_filters(self, query: str) -> Optional[Dict]:
        """Try to extract filters using rules alone. Returns filter dict if the
        entire input maps to known filter patterns, None if LLM parsing is needed."""
        text = query.lower().strip()
        filters: Dict = {}

        # Extract salary patterns first and remove from text
        salary_match = self._SALARY_PATTERN.search(text)
        if salary_match:
            filters["min_salary"] = float(salary_match.group(1)) * 1000
            text = text[:salary_match.start()] + text[salary_match.end():]

        # Match multi-word phrases and remove from text
        for phrase, filt in self._MULTI_WORD_FILTERS:
            if phrase in text:
                filters.update(filt)
                text = text.replace(phrase, " ")

        # Split remaining text, remove filler words, match single-word filters
        words = [w for w in text.split() if w and w not in self._FILLER_WORDS]

        unmatched = []
        for word in words:
            if word in self._SINGLE_WORD_FILTERS:
                filters.update(self._SINGLE_WORD_FILTERS[word])
            else:
                unmatched.append(word)

        # Only succeed if ALL content was consumed and at least one filter found
        if not unmatched and filters:
            return filters

        return None

    def _parse_refinement_intent(
        self,
        state: ConversationState,
        new_query: str,
    ) -> Tuple[Dict, int]:
        """Determine if new query is a refinement, pivot, or other action."""
        context = ""
        if state.last_query:
            context += f"Previous query: {state.last_query}\n"
        active = self._filters_to_string(state.active_filters)
        if active != "none":
            context += f"Active filters: {active}\n"

        prompt = INTENT_PROMPT_TEMPLATE.format(context=context, new_query=new_query)

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
            intent = json.loads(content)
        except (json.JSONDecodeError, IndexError):
            intent = {"type": "pivot", "filters": {}}

        return intent, response.usage.total_tokens

    def _merge_filters(self, existing: SearchFilters, new: Dict) -> SearchFilters:
        """Merge new filters with existing ones. New overrides existing."""
        merged_dict = {k: v for k, v in existing.__dict__.items() if v is not None}

        for key, value in new.items():
            if value is not None and hasattr(SearchFilters, key):
                merged_dict[key] = value

        merged = SearchFilters(**{k: v for k, v in merged_dict.items() if hasattr(SearchFilters, k)})

        # Handle contradictions: remote overrides specific location
        if merged.is_remote and merged.location:
            merged.location = None

        return merged

    def _filters_to_string(self, filters: SearchFilters) -> str:
        """Convert filters to readable string."""
        parts = []
        if filters.is_remote:
            parts.append("remote")
        if filters.seniority:
            parts.append(filters.seniority)
        if filters.location:
            parts.append(filters.location)
        if filters.org_type:
            parts.append(filters.org_type)
        if filters.min_salary:
            parts.append(f"${filters.min_salary/1000:.0f}K+")
        if filters.industry:
            parts.append(filters.industry)
        return ", ".join(parts) if parts else "none"
