"""Small console diagnostics used by pipeline stages."""

import numpy as np


RULE_WIDTH = 88


def print_separator(prefix: str, title: str, char: str = "-") -> None:
    print()
    print(char * RULE_WIDTH)
    print(f"[{prefix}] {title}")
    print(char * RULE_WIDTH)


def print_state_stats(
    state,
    step_name: str,
    context: str | None = None,
    prefix: str = "postprocess",
) -> None:
    n_proteins = int(state.indptr.size - 1)
    head = f"[{prefix}] postprocess"
    if context:
        head += f" | {context}"
    if n_proteins == 0:
        print(f"{head} | {step_name} | proteins=0")
        return
    terms_per_protein = state.indptr[1:] - state.indptr[:-1]
    print(
        f"{head} | {step_name} | proteins={n_proteins:,} | "
        f"terms/protein min={int(terms_per_protein.min())} "
        f"median={float(np.median(terms_per_protein)):.1f} "
        f"max={int(terms_per_protein.max())}"
    )
