"""`python -m matrix`: the CLI entrypoint SP10's burst benchmark lane needs, wiring `matrix.
live_driver`'s real components into the already-built staged runner (`matrix.runner.run_matrix`).

This is the OPERATOR/BURST step (`task matrix:live`), never the PR lane: it makes real live calls
(TEI, OpenAI, Anthropic, Ollama, Postgres) and needs real keys/endpoints. `task test` never reaches
a live call through this module: every function here is either pure argument/config assembly
(hermetically tested, `testing/tests/test_matrix_main.py`) or explicitly gated behind a real
network call this file never fakes.

Spend discipline: stage 3's generator cells are gated by `matrix.spend_gate.check_spend` inside
`matrix.runner.run_matrix` itself (unchanged; this file only supplies real, positive `estimated_usd`
values per `matrix.live_driver.estimate_generation_cost_usd`); the variant-comparison stage has NO
such gate in `run_matrix`, so this entrypoint pre-checks it itself via `matrix.live_driver.
build_variants_config` before ever constructing a paid gateway for it. CROSS STAGE RECONCILIATION:
`build_variants_config` returns an UPDATED `SpendGate` (the admitted variant stage estimate already
recorded into it), and `main()` reassigns its own local `spend_gate` to that returned gate before
calling `run_matrix` -- never the pristine gate constructed at the top of `main()`. Without this,
stage 3's own admission inside `run_matrix` would independently check the SAME starting budget the
variant stage was ALSO just admitted against, and the two checks could each report "fits" while
their combined real spend silently exceeds the hard ceiling. `dropped_cells` (stage 3's own ledger)
and the variant stage's own pre-check refusal (if any) are both printed after the run, never
silently swallowed.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from dataset_tools.manifest import DATASET_VERSION
from rag_tools.chunker import chunker_hash

from matrix.cache import MatrixCache
from matrix.cases import MatrixCase, load_matrix_cases
from matrix.embedders import EmbedderComponent
from matrix.generators import GeneratorComponent
from matrix.live_driver import (
    MissingEnvVarError,
    build_baseline_embedder_components,
    build_bge_m3_embedder_component,
    build_claude_generator_component,
    build_gpt_generator_component,
    build_judge_gateway,
    build_openai_embedder_component,
    build_reranker_components,
    build_variants_config,
    construct_live_pgvector_retriever,
)
from matrix.ollama_generator import build_ollama_generator_component
from matrix.runner import MatrixRunConfig, run_matrix
from matrix.spend_gate import CEILINGS_USD, SpendGate

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CASES_PATH = _REPO_ROOT / "testing" / "harness" / "dataset_tools" / "seed_cases.jsonl"
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "var" / "matrix"
_DEFAULT_CASSETTE_DIR = _DEFAULT_OUTPUT_ROOT / "cassettes"
_DEFAULT_CACHE_DIR = _DEFAULT_OUTPUT_ROOT / "cache"

_VALID_EMBEDDERS = frozenset({"bge-m3", "openai"})
_VALID_GENERATORS = frozenset({"anthropic", "openai", "ollama"})
_VALID_VARIANT_GENERATORS = ("anthropic", "openai", "ollama", "none")

# Three judges from three disjoint provider families (D15's own "3 model cross provider jury",
# `judge.panel.panel_vote`'s own module docstring): the SAME three axes the generator set already
# names, reused for judging too, via `matrix.live_driver.build_judge_gateway` -- never a fourth
# model family this repo has no key for.
_JUDGE_PROVIDERS = ("anthropic", "openai", "ollama")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _validate_choices(values: Sequence[str], valid: frozenset[str], *, flag: str) -> tuple[str, ...]:
    unknown = [v for v in values if v not in valid]
    if unknown:
        raise ValueError(f"{flag}: unknown value(s) {unknown} (valid: {sorted(valid)})")
    return tuple(values)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m matrix",
        description=(
            "Run the SP9 staged benchmark matrix (embedders -> rerankers -> generators, plus the "
            "naive/agentic/graph variant comparison) against REAL components: TEI, OpenAI, "
            "Anthropic, Ollama, Postgres. Operator/burst step; never the PR lane."
        ),
    )
    parser.add_argument("--cases", type=Path, default=_DEFAULT_CASES_PATH, help="Seed-set-shaped JSONL case file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Manifest + per-query output dir (default: var/matrix/<run-id>).")
    parser.add_argument("--run-id", default=None, help="Run identity (default: a timestamped id).")
    parser.add_argument("--git-sha", default=None, help="Default: $GIT_SHA, else `git rev-parse HEAD`.")
    parser.add_argument("--corpus-version", default="corpus-0.1.1")
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--chunker-config-hash", default=None, help="Default: the real, freshly-computed chunker_hash().")
    parser.add_argument("--k-retrieval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--n-top-configs", type=int, default=2)
    parser.add_argument("--reranker-depths", default="20,50,100", help="Comma-separated ints.")
    parser.add_argument("--embedders", default="bge-m3,openai", help="Comma-separated subset of {bge-m3, openai}.")
    parser.add_argument("--generators", default="anthropic,openai,ollama", help="Comma-separated subset of {anthropic, openai, ollama}.")
    parser.add_argument(
        "--variant-generator", default="ollama", choices=_VALID_VARIANT_GENERATORS,
        help="Which generator backs the naive/agentic/graph variant stage, or 'none' to skip it.",
    )
    parser.add_argument("--cassette-dir", type=Path, default=_DEFAULT_CASSETTE_DIR)
    parser.add_argument("--cache-dir", type=Path, default=_DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--avg-output-tokens", type=int, default=300,
        help="Cost-estimate tuning: assumed output tokens per generate call.",
    )
    parser.add_argument("--openai-ceiling", type=float, default=None, help="Default: matrix.spend_gate.CEILINGS_USD['openai'] ($20).")
    parser.add_argument("--anthropic-ceiling", type=float, default=None, help="Default: matrix.spend_gate.CEILINGS_USD['anthropic'] ($10).")
    parser.add_argument("--limit-cases", type=int, default=None, help="Run only the first N cases (a cheap smoke run knob).")
    parser.add_argument("--pool-size", type=int, default=None, help="Default: the widest reranker depth swept.")
    return parser


def resolve_git_sha(explicit: Optional[str]) -> str:
    """`--git-sha` wins; else `$GIT_SHA`; else a real `git rev-parse HEAD` (this is a live/operator
    CLI, never the hermetic lane, so shelling out to the actual checkout's own git_sha here is the
    honest identity, not a determinism violation -- the "no wall clock, no ambient state" rule binds
    the RUNTIME graph's own injected factories, not a one-off benchmark run's own label)."""
    if explicit:
        return explicit
    env_sha = os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=_REPO_ROOT, check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(
            "could not resolve a git_sha: pass --git-sha explicitly, set GIT_SHA, or run this "
            "from inside a git checkout (`git rev-parse HEAD` failed)"
        )
    return result.stdout.strip()


def default_run_id() -> str:
    return "matrix-live-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def build_matrix_run_config(args: argparse.Namespace, *, git_sha: str) -> MatrixRunConfig:
    return MatrixRunConfig(
        run_id=args.run_id or default_run_id(),
        git_sha=git_sha,
        corpus_version=args.corpus_version,
        dataset_version=args.dataset_version,
        chunker_config_hash=args.chunker_config_hash or chunker_hash(),
        k_retrieval=args.k_retrieval,
        seed=args.seed,
        n_top_configs=args.n_top_configs,
        reranker_depths=_parse_int_tuple(args.reranker_depths),
    )


def load_cases_for_run(args: argparse.Namespace) -> tuple[MatrixCase, ...]:
    cases = load_matrix_cases(args.cases)
    if args.limit_cases is not None:
        cases = cases[: args.limit_cases]
    return cases


def _build_embedders(
    embedder_ids: Sequence[str], *, pool_size: int,
) -> list[EmbedderComponent]:  # pragma: no cover - live only, needs Postgres + TEI/OpenAI reachable
    """BM25 + exact_scan are ALWAYS built (D8/research 14: never omitted from a stage 1 run), over
    ONE shared real `PgvectorRetriever` (one live fingerprint check, not one per component); the
    bge-m3/openai axes in `embedder_ids` are added on top."""
    components: list[EmbedderComponent] = []
    shared_retriever = construct_live_pgvector_retriever(
        pg_dsn=None, tei_embed_url=None, tei_rerank_url=None, index_dir=None,
    )
    bm25, exact_scan = build_baseline_embedder_components(pool_size=pool_size, retriever=shared_retriever)
    components.extend([bm25, exact_scan])
    if "bge-m3" in embedder_ids:
        components.append(build_bge_m3_embedder_component(pool_size=pool_size, retriever=shared_retriever))
    if "openai" in embedder_ids:
        components.append(build_openai_embedder_component(pool_size=pool_size))
    return components


def _build_generators(
    generator_ids: Sequence[str], *, cases: Sequence[MatrixCase], cassette_dir: Path, avg_output_tokens: int,
) -> list[GeneratorComponent]:  # pragma: no cover - live only, needs provider keys/daemon
    generators: list[GeneratorComponent] = []
    if "anthropic" in generator_ids:
        generators.append(
            build_claude_generator_component(cases=cases, cassette_dir=cassette_dir, avg_output_tokens=avg_output_tokens)
        )
    if "openai" in generator_ids:
        generators.append(
            build_gpt_generator_component(cases=cases, cassette_dir=cassette_dir, avg_output_tokens=avg_output_tokens)
        )
    if "ollama" in generator_ids:
        generators.append(build_ollama_generator_component(cassette_dir=cassette_dir))
    return generators


def _build_judges(cassette_dir: Path) -> list:  # pragma: no cover - live only, needs provider keys/daemon
    return [build_judge_gateway(provider, cassette_dir=cassette_dir) for provider in _JUDGE_PROVIDERS]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        embedder_ids = _validate_choices(_parse_csv(args.embedders), _VALID_EMBEDDERS, flag="--embedders")
        generator_ids = _validate_choices(_parse_csv(args.generators), _VALID_GENERATORS, flag="--generators")
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cases = load_cases_for_run(args)
    if not cases:
        print("error: no cases loaded (check --cases / --limit-cases)", file=sys.stderr)
        return 2

    try:
        git_sha = resolve_git_sha(args.git_sha)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = build_matrix_run_config(args, git_sha=git_sha)
    pool_size = args.pool_size or max((*config.reranker_depths, config.k_retrieval))
    output_dir = args.output_dir or (_DEFAULT_OUTPUT_ROOT / config.run_id)
    cassette_dir = Path(args.cassette_dir)
    cache_dir = Path(args.cache_dir)

    ceilings = dict(CEILINGS_USD)
    if args.openai_ceiling is not None:
        ceilings["openai"] = args.openai_ceiling
    if args.anthropic_ceiling is not None:
        ceilings["anthropic"] = args.anthropic_ceiling
    spend_gate = SpendGate(ceilings=ceilings)

    try:
        embedders = _build_embedders(embedder_ids, pool_size=pool_size)
        rerankers = list(build_reranker_components())
        generators = _build_generators(
            generator_ids, cases=cases, cassette_dir=cassette_dir, avg_output_tokens=args.avg_output_tokens,
        )
        judges = _build_judges(cassette_dir)
        judge_ids = tuple(f"judge-{provider}" for provider in _JUDGE_PROVIDERS)

        variants = None
        if args.variant_generator != "none":
            # `spend_gate` is reassigned to the RETURNED (reconciled) gate, never left as the
            # pristine one built above -- see the module docstring's "CROSS STAGE RECONCILIATION".
            variants, spend_gate, refusal_reason = build_variants_config(
                cases=cases, gate=spend_gate, provider=args.variant_generator, cassette_dir=cassette_dir,
                avg_output_tokens=args.avg_output_tokens,
            )
            if refusal_reason is not None:
                print(f"variant comparison stage SKIPPED: {refusal_reason}")
    except MissingEnvVarError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    manifest = run_matrix(
        cases, embedders=embedders, rerankers=rerankers, generators=generators,
        judges=judges, judge_ids=judge_ids, cache=MatrixCache(cache_dir), config=config,
        output_dir=output_dir, spend_gate=spend_gate, variants=variants,
    )

    print(f"run_id={config.run_id} git_sha={config.git_sha} wrote manifest to {output_dir / 'manifest.json'}")
    dropped = manifest.get("dropped_cells") or []
    if dropped:
        print(f"dropped_cells ({len(dropped)}):")
        for cell in dropped:
            print(f"  {cell['component_id']} ({cell['provider']}): {cell['reason']}")
    else:
        print("dropped_cells: none")
    return 0


__all__ = ["build_arg_parser", "build_matrix_run_config", "default_run_id", "load_cases_for_run", "main", "resolve_git_sha"]

if __name__ == "__main__":
    raise SystemExit(main())
