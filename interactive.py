#!/usr/bin/env python3
"""Interactive search bar for the AI Job Search System."""

import os
import time

from dotenv import load_dotenv

from src.data_loader import JobDataset
from src.search_engine import SearchEngine
from src.refinement_engine import RefinementEngine, ConversationState


def print_results(results, state, ref_engine, max_display=10):
    """Print search results."""
    filters_str = ref_engine._filters_to_string(state.active_filters)
    print(f"\n  Showing {min(max_display, len(results))} results | Active filters: {filters_str}")
    print(f"  {'─' * 74}")

    for i, result in enumerate(results[:max_display], 1):
        print(f"  {i:>2}. {result.job.to_display_string()}")
        if result.job.skills:
            print(f"      Skills: {', '.join(result.job.skills[:6])}")
    print()


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file")

    print("Loading dataset...")
    t0 = time.time()
    dataset = JobDataset("data/jobs.jsonl")
    print(f"Loaded {len(dataset):,} jobs in {time.time() - t0:.1f}s\n")

    engine = SearchEngine(dataset, api_key)
    ref_engine = RefinementEngine(engine)
    state = ConversationState()

    print("=" * 78)
    print("  AI Job Search")
    print("=" * 78)
    print("  Type a search query to find jobs.")
    print("  Follow up to refine results (e.g. \"make it remote\", \"at startups\").")
    print("  Commands:  /new  reset conversation   /tokens  show usage   /quit  exit")
    print("=" * 78)

    while True:
        try:
            query = input("\n🔍 ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in ("/quit", "/exit", "/q"):
            print("Goodbye!")
            break

        if query.lower() in ("/new", "/reset", "/clear"):
            state = ConversationState()
            print("  Conversation reset. Start a new search.")
            continue

        if query.lower() in ("/tokens", "/usage"):
            print(f"\n{engine.get_token_report()}")
            continue

        t0 = time.time()
        results, state, tokens = ref_engine.refine(state, query, top_k=10)
        elapsed = time.time() - t0

        print(f"  ({tokens} tokens, {elapsed:.1f}s)")
        print_results(results, state, ref_engine)


if __name__ == "__main__":
    main()
