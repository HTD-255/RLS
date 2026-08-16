"""
export_for_paper.py — Export final assets into the paper/ directory.

Collects and organises all outputs needed for the LaTeX paper:
    • Copies figures (PDF) into paper/figures/
    • Generates LaTeX tables into paper/tables/
    • Creates a stub references.bib
    • Generates a paper structure checklist

Usage
-----
    python analysis/export_for_paper.py --data data/ --output paper/
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =========================================================================== #
#  Export figures
# =========================================================================== #

def export_figures(
    figures_dir: str = "figures",
    output_dir: str = "paper/figures",
) -> None:
    """Copy all PDF figures into the paper figures directory."""
    os.makedirs(output_dir, exist_ok=True)
    src = Path(figures_dir)

    if not src.exists():
        print(f"  [!] Figures directory not found: {figures_dir}")
        return

    copied = 0
    for pdf in src.glob("*.pdf"):
        dest = Path(output_dir) / pdf.name
        shutil.copy2(pdf, dest)
        copied += 1
        print(f"    {pdf.name} → {dest}")

    print(f"  ✅ {copied} figures exported to {output_dir}/")


# =========================================================================== #
#  Export tables
# =========================================================================== #

def export_tables(
    data_dir: str = "data",
    output_dir: str = "paper/tables",
) -> None:
    """Generate LaTeX tables and save individually."""
    from analysis.generate_tables import (
        LaTeXFormatter,
        generate_table1,
        generate_table2,
        generate_table3,
        generate_table4,
    )

    os.makedirs(output_dir, exist_ok=True)
    fmt = LaTeXFormatter()

    tables = [
        ("table1_results.tex",      generate_table1),
        ("table2_lambda.tex",       generate_table2),
        ("table3_perturbation.tex", generate_table3),
        ("table4_multisensor.tex",  generate_table4),
    ]

    for filename, gen_fn in tables:
        try:
            content = gen_fn(data_dir, fmt)
            path = os.path.join(output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"    {filename}")
        except Exception as e:
            print(f"    ❌ {filename}: {e}")

    print(f"  ✅ Tables exported to {output_dir}/")


# =========================================================================== #
#  Generate references.bib stub
# =========================================================================== #

def generate_bib(output_path: str = "paper/references.bib") -> None:
    """Create a starter references.bib with key citations."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    bib = r"""% ======================================================================
% References — AutoCar III Steering Self-Calibration
% ======================================================================

% --- Core Methods ---

@book{ljung1999system,
  title     = {System Identification: Theory for the User},
  author    = {Ljung, Lennart},
  year      = {1999},
  edition   = {2nd},
  publisher = {Prentice Hall},
}

@book{haykin2002adaptive,
  title     = {Adaptive Filter Theory},
  author    = {Haykin, Simon},
  year      = {2002},
  edition   = {4th},
  publisher = {Prentice Hall},
  note      = {Ch. 13: Recursive Least Squares},
}

@book{rajamani2012vehicle,
  title     = {Vehicle Dynamics and Control},
  author    = {Rajamani, Rajesh},
  year      = {2012},
  edition   = {2nd},
  publisher = {Springer},
}

@book{rawlings2017mpc,
  title     = {Model Predictive Control: Theory, Computation, and Design},
  author    = {Rawlings, James B. and Mayne, David Q. and Diehl, Moritz},
  year      = {2017},
  edition   = {2nd},
  publisher = {Nob Hill Publishing},
}

% --- Related Work (RLS in Automotive) ---

@article{kim2024parameter,
  title   = {Development of a Parameter-Free Adaptive Steering Control
             Algorithm for Universal Path Tracking of Autonomous Vehicles
             with Recursive Least Squares},
  author  = {Kim, et al.},
  journal = {Transactions of the Korean Society of Automotive Engineers},
  year    = {2024},
}

% --- Platform ---

@manual{hanback2023autocar,
  title        = {AutoCar {III} User Manual \& SDK Documentation},
  organization = {Hanback Electronics},
  year         = {2023},
}

% --- Add your additional references below ---

"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(bib)
    print(f"  ✅ references.bib created at {output_path}")


# =========================================================================== #
#  Generate paper skeleton (main.tex)
# =========================================================================== #

def generate_paper_skeleton(output_path: str = "paper/main.tex") -> None:
    """Create a LaTeX paper skeleton matching the research plan structure."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    tex = r"""\documentclass[conference]{IEEEtran}

\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{algorithm}
\usepackage{algpseudocode}
\usepackage{cite}

\title{Online Self-Calibrating Steering Control Using Recursive Least
       Squares with IMU Feedback on a Small-Scale Autonomous Vehicle}

\author{
  \IEEEauthorblockN{Author Name}
  \IEEEauthorblockA{Affiliation\\
    Email: author@example.com}
}

\begin{document}

\maketitle

\begin{abstract}
% TODO: Write abstract (150--200 words)
Small-scale autonomous vehicles used in education and research suffer from
steering misalignment due to mechanical wear, surface changes, and load
variations. Existing approaches rely on offline linear regression calibration,
which cannot adapt in real time. This paper proposes an online
self-calibrating steering control framework using Recursive Least Squares
(RLS) with exponential forgetting factor and multi-channel IMU feedback. We
compare four methods---offline linear regression, PID, RLS, and lightweight
MPC---on the Hanback AutoCar III platform. Experiments demonstrate that RLS
with $\lambda=0.95$ achieves the lowest yaw error and recovers from
perturbations significantly faster than other methods.
\end{abstract}

\begin{IEEEkeywords}
self-calibration, recursive least squares, IMU, steering control,
autonomous vehicle, online learning
\end{IEEEkeywords}

% ==================================================================
\section{Introduction}
\label{sec:intro}
% TODO: Motivation, limitation of offline calibration, contribution summary

% ==================================================================
\section{Related Work}
\label{sec:related}
% TODO: IMU-based calibration, adaptive steering, RLS in automotive

% ==================================================================
\section{System Description}
\label{sec:system}
% TODO: AutoCar III hardware, IMU specs, existing calibration approach

% ==================================================================
\section{Proposed Approach}
\label{sec:approach}

\subsection{Problem Formulation}
% steer = theta^T x

\subsection{RLS with Forgetting Factor}
% Algorithm 1

\subsection{Multi-Sensor Feature Vector}
% x = [gyro_z, euler_yaw, accel_y, 1]

\subsection{Online Calibration Loop}
% Diagram / pseudocode

% ==================================================================
\section{Experimental Setup}
\label{sec:experiments}

\subsection{Test Environment}
% Track, surface, speed, sampling rate

\subsection{Baseline Methods}
% LR, PID, MPC descriptions

\subsection{Evaluation Metrics}
% MAE, RMSE, convergence time, recovery time

% ==================================================================
\section{Results and Discussion}
\label{sec:results}

\subsection{Straight-Line Calibration (Exp.~1)}
% \input{tables/table1_results}
% \includegraphics[width=\columnwidth]{figures/yaw_error_timeseries}

\subsection{Forgetting Factor Analysis (Exp.~2)}
% \input{tables/table2_lambda}
% \includegraphics[width=\columnwidth]{figures/lambda_analysis}

\subsection{Perturbation Response (Exp.~3)}
% \input{tables/table3_perturbation}
% \includegraphics[width=\columnwidth]{figures/perturbation_response}

\subsection{Multi-Sensor Comparison (Exp.~4)}
% \input{tables/table4_multisensor}

\subsection{Computational Cost Analysis}
% \includegraphics[width=\columnwidth]{figures/boxplot_comparison}

% ==================================================================
\section{Conclusion and Future Work}
\label{sec:conclusion}
% TODO: Summary, limitations, future work

\bibliographystyle{IEEEtran}
\bibliography{references}

\end{document}
"""
    if not os.path.exists(output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(tex)
        print(f"  ✅ Paper skeleton created at {output_path}")
    else:
        print(f"  ⏭  {output_path} already exists (not overwritten)")


# =========================================================================== #
#  Checklist
# =========================================================================== #

def print_paper_checklist(paper_dir: str = "paper") -> None:
    """Print a checklist of required paper assets."""
    print(f"\n{'=' * 60}")
    print("  Paper Assets Checklist")
    print(f"{'=' * 60}")

    required = {
        "main.tex":                  "Paper source",
        "references.bib":           "Bibliography",
        "figures/convergence_curves.pdf":    "Fig 2: RLS convergence",
        "figures/yaw_error_timeseries.pdf":  "Fig 3: Yaw error series",
        "figures/perturbation_response.pdf": "Fig 4: Perturbation ★",
        "figures/lambda_analysis.pdf":       "Fig 5: λ analysis",
        "figures/boxplot_comparison.pdf":     "Fig 6: Box plot",
        "figures/multisensor_comparison.pdf": "Fig 7: Multi-sensor",
        "figures/recovery_comparison.pdf":    "Fig 8: Recovery bars",
        "tables/table1_results.tex":         "Table 1: Results",
        "tables/table2_lambda.tex":          "Table 2: λ sweep",
        "tables/table3_perturbation.tex":    "Table 3: Perturbation",
        "tables/table4_multisensor.tex":     "Table 4: Multi-sensor",
    }

    found = 0
    missing = 0
    for rel_path, desc in required.items():
        full = os.path.join(paper_dir, rel_path)
        exists = os.path.exists(full)
        icon = "✅" if exists else "❌"
        if exists:
            found += 1
        else:
            missing += 1
        print(f"  {icon} {rel_path:<45} {desc}")

    print(f"\n  {found}/{found + missing} assets ready "
          f"({'🎉 All done!' if missing == 0 else f'{missing} missing'})")


# =========================================================================== #
#  Master export
# =========================================================================== #

def export_all(
    data_dir: str = "data",
    figures_dir: str = "figures",
    paper_dir: str = "paper",
) -> None:
    """Run the full export pipeline."""
    print("\n" + "╔" + "═" * 56 + "╗")
    print("║  Exporting Paper Assets                                 ║")
    print("╚" + "═" * 56 + "╝\n")

    print("▶ Step 1: Paper skeleton")
    generate_paper_skeleton(os.path.join(paper_dir, "main.tex"))

    print("\n▶ Step 2: References")
    generate_bib(os.path.join(paper_dir, "references.bib"))

    print("\n▶ Step 3: Figures")
    export_figures(figures_dir, os.path.join(paper_dir, "figures"))

    print("\n▶ Step 4: Tables")
    export_tables(data_dir, os.path.join(paper_dir, "tables"))

    print("\n▶ Step 5: Checklist")
    print_paper_checklist(paper_dir)


# =========================================================================== #
#  CLI
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export final paper assets"
    )
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--figures", type=str, default="figures")
    parser.add_argument("--output", type=str, default="paper")
    args = parser.parse_args()

    export_all(args.data, args.figures, args.output)


if __name__ == "__main__":
    main()
