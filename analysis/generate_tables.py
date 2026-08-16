"""
generate_tables.py — Generate formatted result tables for the paper.

Produces:
    1. Table 1: Quantitative Results Summary  (LaTeX + Markdown)
    2. Table 2: Forgetting Factor Sweep       (LaTeX + Markdown)
    3. Table 3: Perturbation Response          (LaTeX + Markdown)
    4. Table 4: Multi-Sensor Comparison        (LaTeX + Markdown)

Usage
-----
    python analysis/generate_tables.py --data data/
    python analysis/generate_tables.py --data data/ --format latex
    python analysis/generate_tables.py --data data/ --format markdown
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =========================================================================== #
#  Table formatters
# =========================================================================== #

class TableFormatter:
    """Base class for table output formatting."""

    def format(self, headers: List[str], rows: List[List[str]],
               caption: str = "", label: str = "") -> str:
        raise NotImplementedError


class MarkdownFormatter(TableFormatter):
    def format(self, headers: List[str], rows: List[List[str]],
               caption: str = "", label: str = "") -> str:
        lines = []
        if caption:
            lines.append(f"### {caption}\n")

        # Header
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---:" if i > 0 else ":---"
                                      for i in range(len(headers))]) + "|")
        # Rows
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


class LaTeXFormatter(TableFormatter):
    def format(self, headers: List[str], rows: List[List[str]],
               caption: str = "", label: str = "") -> str:
        n_cols = len(headers)
        col_spec = "l" + "r" * (n_cols - 1)

        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{caption}}}" if caption else "",
            f"\\label{{{label}}}" if label else "",
            f"\\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
            " & ".join(f"\\textbf{{{h}}}" for h in headers) + r" \\",
            r"\midrule",
        ]

        for row in rows:
            # Escape special characters
            escaped = [cell.replace("±", r"$\pm$").replace("★", r"$\star$")
                       for cell in row]
            lines.append(" & ".join(escaped) + r" \\")

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        return "\n".join(line for line in lines if line)


def get_formatter(fmt: str) -> TableFormatter:
    if fmt == "latex":
        return LaTeXFormatter()
    return MarkdownFormatter()


# =========================================================================== #
#  Table 1: Quantitative Results Summary (Exp 1)
# =========================================================================== #

def generate_table1(
    data_dir: str,
    fmt: TableFormatter,
) -> str:
    """Main results table comparing all four controllers."""
    summary_path = os.path.join(data_dir, "exp1", "exp1_summary.json")
    if not os.path.exists(summary_path):
        return "  [!] No exp1_summary.json found."

    with open(summary_path) as f:
        summary = json.load(f)

    headers = ["Method", "MAE_ψ (°)", "RMSE_ψ (°)", "T_comp (ms)",
               "Adaptable"]

    rows = []
    for ctrl in ["LR", "PID", "RLS", "MPC"]:
        if ctrl not in summary:
            continue
        d = summary[ctrl]
        mae = f"{d['mae_yaw_mean']:.3f} ± {d['mae_yaw_std']:.3f}"
        rmse = f"{d['rmse_yaw_mean']:.3f}"
        comp = f"{d['compute_ms_mean']:.3f}"
        adapt = "No" if ctrl in ["LR", "PID"] else "Yes"
        prefix = "★ " if ctrl == "RLS" else ""
        rows.append([f"{prefix}{ctrl}", mae, rmse, comp, adapt])

    return fmt.format(
        headers, rows,
        caption="Quantitative comparison of steering controllers (Exp. 1)",
        label="tab:exp1-results",
    )


# =========================================================================== #
#  Table 2: Forgetting Factor Sweep (Exp 2)
# =========================================================================== #

def generate_table2(
    data_dir: str,
    fmt: TableFormatter,
) -> str:
    """λ sweep results table."""
    summary_path = os.path.join(data_dir, "exp2", "exp2_summary.json")
    if not os.path.exists(summary_path):
        return "  [!] No exp2_summary.json found."

    with open(summary_path) as f:
        summary = json.load(f)

    headers = ["λ", "MAE_ψ (°)", "RMSE_ψ (°)", "T_conv (s)",
               "Conv. Rate"]

    # Find best λ
    best_lam = min(summary.items(),
                   key=lambda x: x[1]["mae_yaw_mean"])[0]

    rows = []
    for label in sorted(summary.keys()):
        d = summary[label]
        lam_str = f"{d['lambda']:.2f}"
        mae = f"{d['mae_yaw_mean']:.3f} ± {d['mae_yaw_std']:.3f}"
        rmse = f"{d['rmse_yaw_mean']:.3f}"
        conv = f"{d['convergence_time_mean']:.2f}" if d.get("convergence_time_mean") else "N/C"
        rate = f"{d['convergence_rate']*100:.0f}%"

        prefix = "★ " if label == best_lam else ""
        rows.append([f"{prefix}{lam_str}", mae, rmse, conv, rate])

    return fmt.format(
        headers, rows,
        caption="Effect of forgetting factor $\\lambda$ on RLS performance (Exp. 2)",
        label="tab:exp2-lambda",
    )


# =========================================================================== #
#  Table 3: Perturbation Response (Exp 3)
# =========================================================================== #

def generate_table3(
    data_dir: str,
    fmt: TableFormatter,
) -> str:
    """Perturbation response comparison table."""
    summary_path = os.path.join(data_dir, "exp3", "exp3_summary.json")
    if not os.path.exists(summary_path):
        return "  [!] No exp3_summary.json found."

    with open(summary_path) as f:
        summary = json.load(f)

    headers = ["Method", "MAE_ψ (°)", "Peak_ψ (°)", "T_recovery (s)",
               "Recovery Rate"]

    rows = []
    for ctrl in ["LR", "PID", "RLS", "MPC"]:
        if ctrl not in summary:
            continue
        d = summary[ctrl]
        mae = f"{d['mae_yaw_mean']:.3f} ± {d['mae_yaw_std']:.3f}"
        peak = f"{d['peak_yaw_mean']:.2f}"
        rec = f"{d['recovery_time_mean']:.2f}" if d.get("recovery_time_mean") else "N/R"
        rate = f"{d['recovery_rate']*100:.0f}%"

        prefix = "★ " if ctrl == "RLS" else ""
        rows.append([f"{prefix}{ctrl}", mae, peak, rec, rate])

    return fmt.format(
        headers, rows,
        caption="Perturbation response comparison (Exp. 3)",
        label="tab:exp3-perturbation",
    )


# =========================================================================== #
#  Table 4: Multi-Sensor Comparison (Exp 4)
# =========================================================================== #

def generate_table4(
    data_dir: str,
    fmt: TableFormatter,
) -> str:
    """Multi-sensor configuration comparison table."""
    summary_path = os.path.join(data_dir, "exp4", "exp4_summary.json")
    if not os.path.exists(summary_path):
        return "  [!] No exp4_summary.json found."

    with open(summary_path) as f:
        summary = json.load(f)

    headers = ["Config", "Features", "MAE_ψ (°)", "RMSE_ψ (°)", "T_conv (s)"]

    # Find best
    best_cfg = min(summary.items(),
                   key=lambda x: x[1]["mae_yaw_mean"])[0]

    rows = []
    for cfg, d in summary.items():
        n_feat = str(d.get("n_features", "?"))
        mae = f"{d['mae_yaw_mean']:.3f} ± {d['mae_yaw_std']:.3f}"
        rmse = f"{d['rmse_yaw_mean']:.3f}"
        conv = f"{d['convergence_time_mean']:.2f}" if d.get("convergence_time_mean") else "N/C"

        prefix = "★ " if cfg == best_cfg else ""
        rows.append([f"{prefix}{cfg}", n_feat, mae, rmse, conv])

    return fmt.format(
        headers, rows,
        caption="Multi-sensor fusion comparison with RLS (Exp. 4)",
        label="tab:exp4-multisensor",
    )


# =========================================================================== #
#  Master runner
# =========================================================================== #

def generate_all_tables(
    data_dir: str,
    format_name: str = "markdown",
    output_dir: str = "analysis",
) -> None:
    """Generate all tables and save to file."""
    fmt = get_formatter(format_name)
    ext = ".tex" if format_name == "latex" else ".md"

    tables = {
        "table1_results":      ("Table 1: Results Summary", generate_table1),
        "table2_lambda":       ("Table 2: λ Sweep", generate_table2),
        "table3_perturbation": ("Table 3: Perturbation", generate_table3),
        "table4_multisensor":  ("Table 4: Multi-Sensor", generate_table4),
    }

    print("=" * 50)
    print(f"  Generating Tables ({format_name})")
    print("=" * 50)

    all_output = []

    for filename, (title, gen_fn) in tables.items():
        print(f"\n▶ {title}")
        try:
            table_str = gen_fn(data_dir, fmt)
            print(table_str)
            all_output.append(f"## {title}\n\n{table_str}\n")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            all_output.append(f"## {title}\n\n  Error: {e}\n")

    # Save combined file
    os.makedirs(output_dir, exist_ok=True)
    combined_path = os.path.join(output_dir, f"all_tables{ext}")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(all_output))

    print(f"\n✅ All tables saved to {combined_path}")


# =========================================================================== #
#  CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate formatted result tables"
    )
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--format", type=str, default="both",
                        choices=["markdown", "latex", "both"])
    parser.add_argument("--output", type=str, default="analysis")
    args = parser.parse_args()

    if args.format == "both":
        generate_all_tables(args.data, "markdown", args.output)
        print("\n")
        generate_all_tables(args.data, "latex", args.output)
    else:
        generate_all_tables(args.data, args.format, args.output)


if __name__ == "__main__":
    main()
