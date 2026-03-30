#!/usr/bin/env python3
import argparse
from collections import Counter

from rewardbench import load_eval_dataset
from rewardbench.constants import EXAMPLE_COUNTS, SUBSET_MAPPING
from rich.console import Console
from rich.table import Table


def parse_args():
    parser = argparse.ArgumentParser(description="Print sample counts for each RewardBench subset.")
    parser.add_argument(
        "--pref_sets",
        action="store_true",
        help="Use preference test sets instead of the core RewardBench set.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=None,
        help="Optional max turns filter passed to load_eval_dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    console = Console()

    # Use custom formatting to avoid requiring a tokenizer/chat template.
    _, subsets = load_eval_dataset(
        core_set=not args.pref_sets,
        custom_dialogue_formatting=True,
        keep_columns=["subset"],
        max_turns=args.max_turns,
    )

    counts = Counter(subsets)
    total = sum(counts.values())
    mode = "pref_sets" if args.pref_sets else "core_set"

    console.rule("[bold cyan]RewardBench Subset Sizes[/bold cyan]")
    console.print(f"[bold]Mode:[/bold] {mode}")
    console.print(f"[bold]Total examples:[/bold] {total}")

    subset_table = Table(title="Per-subset counts", show_lines=False)
    subset_table.add_column("Subset", style="bold")
    subset_table.add_column("Count", justify="right", style="green")
    for subset_name in sorted(counts):
        subset_table.add_row(subset_name, str(counts[subset_name]))
    console.print(subset_table)

    if not args.pref_sets:
        section_table = Table(title="Leaderboard section counts", show_lines=False)
        section_table.add_column("Section", style="bold magenta")
        section_table.add_column("Actual count", justify="right", style="green")
        section_table.add_column("Expected count", justify="right", style="yellow")
        section_table.add_column("Subsets", style="dim")

        for section, section_subsets in SUBSET_MAPPING.items():
            actual_count = sum(counts.get(subset, 0) for subset in section_subsets)
            expected_count = sum(EXAMPLE_COUNTS.get(subset, 0) for subset in section_subsets)
            section_table.add_row(
                section,
                str(actual_count),
                str(expected_count),
                ", ".join(section_subsets),
            )
        console.print(section_table)


if __name__ == "__main__":
    main()
