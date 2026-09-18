"""Shared aggregate-figure launcher for logKd experiment wrappers."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path


_FIGURE_SUFFIXES = {".png"}


def generate_aggregate_figure(
    batch_dir: Path,
    *,
    plot_type: str,
    input_relative_path: Path,
    output_stem: str,
    output_subdirectory: Path | None = None,
    include_retention: bool = False,
    metrics: Sequence[str] = (),
    group_column: str | None = None,
    groups: Sequence[str] = (),
    where: Mapping[str, str] | None = None,
    title: str | None = None,
    target_label: str | None = None,
) -> list[str]:
    """Run the plot CLI on one aggregate summary and return saved figure paths."""

    input_path = batch_dir / input_relative_path
    if not input_path.exists():
        raise RuntimeError(f"Aggregate plot input was not written: {input_path}")

    subdirectory = Path(output_subdirectory) if output_subdirectory else Path()
    if subdirectory.is_absolute() or ".." in subdirectory.parts:
        raise ValueError(
            "The figure output subdirectory must be a relative path within figures/."
        )
    figures_dir = batch_dir / "figures" / subdirectory
    log_prefix = "_".join((*subdirectory.parts, output_stem))
    log_path = batch_dir / "audit" / f"plot_{log_prefix}_console_output.txt"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"WARNING: Could not create plot-log directory {log_path.parent}: {error}", file=sys.stderr)

    plot_script = Path(__file__).parents[1] / "plot_logkd_results.py"
    command = [
        sys.executable,
        "-u",
        str(plot_script),
        plot_type,
        "--input",
        str(input_path),
        "--output-dir",
        str(figures_dir),
        "--output-stem",
        output_stem,
    ]
    if include_retention:
        command.append("--include-retention")
    if metrics:
        command.extend(("--metrics", *metrics))
    if group_column:
        command.extend(("--group-column", group_column))
    if groups:
        # The predicted-versus-actual CLI creates one panel per scenario.  It
        # accepts ``--scenario-order`` rather than comparison plots' ``--groups``.
        group_argument = "--scenario-order" if plot_type == "predicted-vs-actual" else "--groups"
        command.extend((group_argument, *groups))
    if where:
        for column, value in where.items():
            command.extend(("--where", f"{column}={value}"))
    if title:
        command.extend(("--title", title))
    if target_label:
        command.extend(("--target-label", target_label))

    result = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    output = result.stdout or ""
    try:
        log_path.write_text(output, encoding="utf-8")
    except OSError as error:
        print(f"WARNING: Could not write plot log {log_path}: {error}", file=sys.stderr)
    if result.returncode != 0:
        detail = output.strip()
        detail = f" Output: {detail}" if detail else ""
        raise RuntimeError(
            f"Figure generation failed with exit code {result.returncode}; "
            f"see {log_path}.{detail}"
        )

    saved_paths = [
        str(path)
        for path in sorted(figures_dir.glob(f"{output_stem}*"))
        if path.suffix.lower() in _FIGURE_SUFFIXES
    ]
    if not saved_paths:
        raise RuntimeError(
            f"Figure generation completed without creating figures; see {log_path}."
        )
    for path in saved_paths:
        print(f"Saved figure: {path}", flush=True)
    return saved_paths
