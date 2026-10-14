# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Summarize and cross-check PatternRewriterBenchmark CSV measurements."""

import argparse
import csv
import statistics
from collections import defaultdict


def median(rows, field):
    return statistics.median(float(row[field]) for row in rows)


def summarize(path):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = tuple(int(row[field]) for field in ("layers", "nodes", "rules", "workload", "indexed"))
            groups[key].append(row)

    expected = {
        (layers, rules, workload, indexed)
        for layers in (1, 8, 32)
        for rules in (1, 10, 100, 300, 500)
        for workload in range(4)
        for indexed in (0, 1)
    }
    observed = {(key[0], key[2], key[3], key[4]) for key in groups}
    if observed != expected:
        raise ValueError(f"Incomplete benchmark matrix: missing={expected - observed}, extra={observed - expected}")

    print(
        "| Layers | Nodes | Rules | Workload | Indexed | Prepare median ms | Engine median ms | Engine max ms "
        "| Resolve median ms | Checks | Guards | Rewrites | Output nodes |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for key, rows in sorted(groups.items()):
        if sorted(int(row["repeat"]) for row in rows) != list(range(9)):
            raise ValueError(f"Expected exactly nine repeats: {key}")
        for field in ("visited", "candidates", "structural_checks", "guards", "matches", "rewrites", "output_nodes"):
            if len({row[field] for row in rows}) != 1:
                raise ValueError(f"Nondeterministic {field}: {key}")
        other_key = (*key[:-1], 1 - key[-1])
        other_rows = groups[other_key]
        for field in ("visited", "matches", "rewrites", "output_nodes"):
            if rows[0][field] != other_rows[0][field]:
                raise ValueError(f"Baseline/indexed disagreement in {field}: {key}")

        print(
            f"| {' | '.join(str(value) for value in key)} | {median(rows, 'prepare_ms'):.4f} "
            f"| {median(rows, 'engine_ms'):.4f} | {max(float(row['engine_ms']) for row in rows):.4f} "
            f"| {median(rows, 'resolve_ms'):.4f} | {rows[0]['structural_checks']} | {rows[0]['guards']} "
            f"| {rows[0]['rewrites']} | {rows[0]['output_nodes']} |"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    summarize(parser.parse_args().csv)
