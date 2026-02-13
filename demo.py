#!/usr/bin/env python3
"""Demo script showcasing the AI Job Search System with token tracking."""

import os
import time

from dotenv import load_dotenv

from src.data_loader import JobDataset
from src.search_engine import SearchEngine
from src.refinement_engine import RefinementEngine, ConversationState


def print_results(results, query, tokens, turn_label="", max_display=5):
    """Print search results."""
    print(f"\n{'='*80}")
    if turn_label:
        print(f"{turn_label}")
    print(f'Query: "{query}"')
    print(f"Tokens used: {tokens}")
    print(f"Results: {len(results)} (showing top {min(max_display, len(results))})")
    print(f"{'='*80}")

    for i, result in enumerate(results[:max_display], 1):
        print(f"  {i}. {result.job.to_display_string()}")
        if result.job.skills:
            print(f"     Skills: {', '.join(result.job.skills[:6])}")


def demo_search(engine):
    """Demonstrate single-turn searches."""
    print("\n" + "#" * 80)
    print("# DEMO 1: SINGLE-TURN SEARCHES")
    print("#" * 80)

    queries = [
        "senior software engineer remote",
        "data science internships",
        "product manager roles at early stage startups",
        "backend engineer jobs in New York paying over 150k",
        "entry level design roles",
        "mission-driven nonprofit data roles",
    ]

    total_tokens = 0

    for query in queries:
        results, tokens = engine.search(query, top_k=10)
        print_results(results, query, tokens)
        total_tokens += tokens
        print(f"  Running total: {total_tokens} tokens")

    return total_tokens


def demo_refinement(engine):
    """Demonstrate multi-turn refinement."""
    print("\n" + "#" * 80)
    print("# DEMO 2: MULTI-TURN REFINEMENT FLOWS")
    print("#" * 80)

    ref_engine = RefinementEngine(engine)
    total_tokens = 0

    # Flow 1
    print("\n" + "-" * 80)
    print("Flow 1: Data Science -> Social Good -> Remote")
    print("-" * 80)

    state = ConversationState()
    for i, query in enumerate(
        ["data science jobs", "at companies or non-profits that care about social good", "make it remote"],
        1,
    ):
        results, state, tokens = ref_engine.refine(state, query, top_k=10)
        print_results(results, query, tokens, turn_label=f"Turn {i}")
        print(f"  Active filters: {ref_engine._filters_to_string(state.active_filters)}")
        total_tokens += tokens

    # Flow 2
    print("\n" + "-" * 80)
    print("Flow 2: ML Engineer -> Startups -> Senior")
    print("-" * 80)

    state = ConversationState()
    for i, query in enumerate(
        ["machine learning engineer", "at early stage startups", "senior level only"],
        1,
    ):
        results, state, tokens = ref_engine.refine(state, query, top_k=10)
        print_results(results, query, tokens, turn_label=f"Turn {i}")
        print(f"  Active filters: {ref_engine._filters_to_string(state.active_filters)}")
        total_tokens += tokens

    return total_tokens


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

    search_tokens = demo_search(engine)
    refinement_tokens = demo_refinement(engine)

    total = search_tokens + refinement_tokens

    # Final report
    print("\n" + "=" * 80)
    print("FINAL TOKEN USAGE REPORT")
    print("=" * 80)
    print(f"\nSingle-turn searches (6 queries): {search_tokens:,} tokens")
    print(f"Multi-turn refinements (6 turns):  {refinement_tokens:,} tokens")
    print(f"Total demo tokens:                 {total:,} tokens")
    print(f"\nEstimated cost (GPT-4o-mini): ${total * 0.15 / 1_000_000:.4f}")
    print(f"Average per single-turn search:    {search_tokens // 6} tokens")
    print(f"Average per refinement turn:       {refinement_tokens // 6} tokens")
    print(f"\n{engine.get_token_report()}")

    # Save report
    with open("tokens_report.md", "w") as f:
        f.write("# Token Usage Report\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Total tokens used in demo:** {total:,}\n")
        f.write(f"- **Estimated cost:** ${total * 0.15 / 1_000_000:.4f}\n")
        f.write(f"- **Average per search:** {search_tokens // 6} tokens\n")
        f.write(f"- **Average per refinement:** {refinement_tokens // 6} tokens\n\n")
        f.write("## Detailed Breakdown\n\n")
        f.write(engine.get_token_report())

    print("Token report saved to tokens_report.md")


if __name__ == "__main__":
    main()
