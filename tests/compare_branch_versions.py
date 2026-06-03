#!/usr/bin/env python3
"""
Run side-by-side validation for two PywrDRB checkouts/branches.

What it does:
1) Runs a selected pytest command in each repo.
2) Runs STARFIT legacy-vs-parametric comparison for selected inflow scenarios.
3) Writes a compact markdown summary for apples-to-apples branch review.
"""

import argparse
import os
import subprocess
from datetime import datetime


def _run(cmd, cwd):
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return {
        "cmd": cmd,
        "cwd": cwd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _safe_branch_and_sha(repo_path):
    b = _run("git rev-parse --abbrev-ref HEAD", repo_path)
    s = _run("git rev-parse --short HEAD", repo_path)
    branch = b["stdout"].strip() if b["returncode"] == 0 else "UNKNOWN"
    sha = s["stdout"].strip() if s["returncode"] == 0 else "UNKNOWN"
    return branch, sha


def _label(repo_path):
    return os.path.basename(os.path.abspath(repo_path.rstrip("/")))


def _write_report(path, title, rows):
    with open(path, "w") as f:
        f.write("# {0}\n\n".format(title))
        f.write("Generated: {0}\n\n".format(datetime.utcnow().isoformat() + "Z"))
        for row in rows:
            f.write("## {0}\n\n".format(row["section"]))
            f.write("- status: `{0}`\n".format("PASS" if row["returncode"] == 0 else "FAIL"))
            f.write("- command: `{0}`\n".format(row["cmd"]))
            f.write("- cwd: `{0}`\n\n".format(row["cwd"]))
            if row["stdout"]:
                f.write("stdout:\n\n```\n{0}\n```\n\n".format(row["stdout"][-8000:]))
            if row["stderr"]:
                f.write("stderr:\n\n```\n{0}\n```\n\n".format(row["stderr"][-8000:]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-a", required=True, help="Path to current/feature repo.")
    parser.add_argument("--repo-b", required=True, help="Path to comparison repo (e.g., v2.2 dev).")
    parser.add_argument(
        "--inflow-types",
        default="nhmv10_withObsScaled,nwmv21_withObsScaled",
        help="Comma-separated inflow scenarios for STARFIT comparisons.",
    )
    parser.add_argument(
        "--pytest-cmd",
        default=(
            "python3 -m pytest -q "
            "tests/test_parametric_release_loading.py "
            "tests/test_starfit_parametric_consistency.py"
        ),
        help="Pytest command to run in each repo.",
    )
    parser.add_argument(
        "--output-dir",
        default="model_runs/branch_comparison",
        help="Output directory (created under current working directory).",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    inflow_types = [x.strip() for x in args.inflow_types.split(",") if x.strip()]

    branch_a, sha_a = _safe_branch_and_sha(args.repo_a)
    branch_b, sha_b = _safe_branch_and_sha(args.repo_b)
    label_a = _label(args.repo_a)
    label_b = _label(args.repo_b)

    rows = []

    # Test suites
    for repo_path, label, branch, sha in [
        (args.repo_a, label_a, branch_a, sha_a),
        (args.repo_b, label_b, branch_b, sha_b),
    ]:
        res = _run(args.pytest_cmd, repo_path)
        res["section"] = "Pytest: {0} ({1} @ {2})".format(label, branch, sha)
        rows.append(res)

    # STARFIT implementation comparisons by inflow scenario
    for inflow in inflow_types:
        for repo_path, label, branch, sha in [
            (args.repo_a, label_a, branch_a, sha_a),
            (args.repo_b, label_b, branch_b, sha_b),
        ]:
            cmd = (
                "python3 scripts/compare_starfit_parametric.py "
                "--inflow-type {inflow} "
                "--policy-id default "
                "--work-dir model_runs/starfit_comparison_{label}_{inflow}"
            ).format(inflow=inflow, label=label)
            res = _run(cmd, repo_path)
            res["section"] = "STARFIT compare: {0} {1} ({2} @ {3})".format(
                inflow, label, branch, sha
            )
            rows.append(res)

    report_path = os.path.join(output_dir, "branch_version_comparison_report.md")
    _write_report(
        report_path,
        "Branch Version Comparison ({0} vs {1})".format(label_a, label_b),
        rows,
    )

    print("Wrote report: {0}".format(report_path))
    print("Repo A: {0} ({1} @ {2})".format(label_a, branch_a, sha_a))
    print("Repo B: {0} ({1} @ {2})".format(label_b, branch_b, sha_b))


if __name__ == "__main__":
    main()
