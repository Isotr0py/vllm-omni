#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render a RESULTS.md comparison table from a bench output directory.

Usage: render_results.py <bench-dir>
"""

import glob
import json
import os
import sys
from datetime import datetime

KEYS = [
    ("completed", "completed"),
    ("request_throughput", "req/s"),
    ("output_token_throughput", "out tok/s"),
    ("mean_ttft_ms", "TTFT mean ms"),
    ("p99_ttft_ms", "TTFT p99 ms"),
    ("mean_itl_ms", "ITL mean ms"),
    ("p99_itl_ms", "ITL p99 ms"),
]


def main() -> None:
    outdir = sys.argv[1]
    cpu: dict[tuple[str, str], float] = {}
    summary = os.path.join(outdir, "summary.txt")
    if os.path.exists(summary):
        for line in open(summary):
            parts = line.split()
            label, rate = parts[0], parts[1].split("=")[1]
            cpu_s = [p.split("=")[1] for p in parts if p.startswith("frontend_cpu_s")]
            if cpu_s:
                cpu[(label, rate)] = float(cpu_s[0])

    rows: dict[str, dict[str, dict]] = {}
    for path in sorted(glob.glob(os.path.join(outdir, "*_rate*.json"))):
        name = os.path.basename(path)
        label, rate = name.split("_rate")
        rate = rate.removesuffix(".json")
        d = json.load(open(path))
        d["frontend_tree_cpu_s"] = cpu.get((label, rate))
        rows.setdefault(rate, {})[label] = d

    lines = [
        "# vllm-omni-rs vs Python frontend — e2e benchmark",
        "",
        f"- bench dir: `{outdir}`",
        f"- rendered: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "Per request-rate row, Python = stock omni api_server, Rust = vllm-omni-rs + façade.",
        "`frontend_tree_cpu_s` sums user+sys CPU seconds over the frontend root process tree",
        "(Rust: vllm-omni-rs + façade + stage procs; Python: api_server + stage procs).",
        "",
    ]
    for rate, pair in rows.items():
        lines.append(f"## request-rate = {rate}")
        lines.append("")
        header = ["metric", *pair.keys()]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "---|" * len(header))
        for key, title in KEYS + [("frontend_tree_cpu_s", "frontend tree CPU s")]:
            cells = []
            for d in pair.values():
                v = d.get(key)
                cells.append(f"{v:.2f}" if isinstance(v, (int, float)) else "-")
            lines.append("| " + title + " | " + " | ".join(cells) + " |")
        lines.append("")

    out = os.path.join(outdir, "RESULTS.md")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(out)


if __name__ == "__main__":
    main()
