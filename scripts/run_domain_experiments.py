"""Run the news/wiki domain grids and export experiment-only comparison tables."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path

import _bootstrap  # noqa: F401

from asr_rescoring.analysis import audit_run
from asr_rescoring.config import PROJECT_ROOT, file_sha256, load_config, resolve_path, write_json
from asr_rescoring.corpus import prepare_corpus
from asr_rescoring.experiments import configure_logging, run_experiments

STUDIES = ("news_only", "wiki_only", "news_wiki")
LABELS = {
    "all_five": "Original five-domain mixture",
    "news_only": "News only",
    "wiki_only": "Wikipedia only",
    "news_wiki": "News + Wikipedia",
}


def export_comparison(prefix: str, reference_run: str) -> Path:
    """Audit new runs and compare all 36 models without rewriting the original run."""
    base = load_config()
    original_path = resolve_path(base, "metrics") / f"{reference_run}_summary.csv"
    original_hash = file_sha256(original_path)
    rows = []
    configs = {"all_five": base} | {
        name: load_config(PROJECT_ROOT / "configs" / f"{name}.yaml") for name in STUDIES
    }
    run_ids = {"all_five": reference_run} | {name: f"{prefix}_{name}" for name in STUDIES}
    analyses = {}
    for name, config in configs.items():
        run_id = run_ids[name]
        if name == "all_five":
            analysis = json.loads(
                (resolve_path(config, "metrics") / f"{run_id}_analysis.json").read_text(
                    encoding="utf-8"
                )
            )
        else:
            analysis = audit_run(config, run_id)
        analyses[name] = analysis
        metric_paths = sorted(resolve_path(config, "metrics").glob(f"{run_id}_order_*.json"))
        if len(metric_paths) != 9:
            raise ValueError(f"{run_id}: expected nine model results")
        for path in metric_paths:
            result = json.loads(path.read_text(encoding="utf-8"))
            for key in ("seed", "scoring", "evaluation", "experiments"):
                if result["config"][key] != base[key]:
                    raise ValueError(f"Incomparable convention: {run_id} {key}")
            original_norm = {
                k: v for k, v in base["preprocessing"].items() if k != "sentences_per_domain"
            }
            current_norm = {
                k: v
                for k, v in result["config"]["preprocessing"].items()
                if k != "sentences_per_domain"
            }
            if original_norm != current_norm:
                raise ValueError(f"Incomparable preprocessing: {run_id}")
            if result["evaluation_sha256"] != file_sha256(resolve_path(base, "evaluation")):
                raise ValueError(f"Different evaluation file: {run_id}")
            tie = analysis["tie_sensitivity"][f"{result['order']}gram_{result['budget']}"]
            rows.append(
                {
                    "corpus": name,
                    "run_id": run_id,
                    "model_id": result["model_id"],
                    "order": result["order"],
                    "budget": result["budget"],
                    "actual_tokens": result["actual_tokens"],
                    **result["metrics"],
                    "training_seconds": result["training_seconds"],
                    "binary_bytes": result["binary_bytes"],
                    "random_top_tie_accuracy": tie["random_tie_accuracy"],
                    "minimum_top_tie_accuracy": tie["minimum_accuracy"],
                    "distinct_text_top_ties": tie["top_ties_between_distinct_normalized_texts"],
                }
            )
    rows.sort(key=lambda r: (list(configs).index(r["corpus"]), r["budget"], r["order"]))
    output_csv = resolve_path(base, "metrics") / f"{prefix}_domain_comparison.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    index = {(r["corpus"], r["budget"], r["order"]): r for r in rows}
    budgets = base["experiments"]["corpus_word_tokens"]
    orders = base["experiments"]["ngram_orders"]
    labels = {100000: "100K", 1000000: "1M", 10000000: "10M"}
    blocks = [
        "<h1>News and Wikipedia: experiment comparison</h1>",
        "<p>This is a report on three domain experiments, not the final assignment report.</p>",
        "<p>Each experiment trains orders 1, 2 and 3 at 100K, 1M and 10M word budgets. "
        "The 500 evaluation records, seed 42, normalization, perplexity denominator, "
        "BOS/EOS and first-index tie rule are fixed. All 27 new models were evaluated. "
        "The original nine models are retained as a reference.</p>",
        "<p><strong>Interpretation:</strong> candidate 0 is always correct, so the "
        "first-candidate baseline is 100%. Seeded random accuracy is 15.6%. "
        "Random top-tie accuracy below is an analytical expectation from saved equal "
        "top scores; official predictions have not been changed.</p>",
        "<p>Enlarged caches retain up to 750,000 normalized unique sentences per domain, "
        "using the same seeded hash-priority rule. Smaller subsets remain nested "
        "whole-sentence prefixes. News+Wikipedia uses round-robin sentence mixing "
        "and cross-domain deduplication; actual word shares are recorded below.</p>",
    ]

    def add_table(headers: list[str], values: list[list[str]]) -> None:
        blocks.append('<div class="scroll"><table><thead><tr>')
        blocks.extend(f"<th>{escape(v)}</th>" for v in headers)
        blocks.append("</tr></thead><tbody>")
        for row in values:
            blocks.append("<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>")
        blocks.append("</tbody></table></div>")

    for name in STUDIES:
        blocks.append(f"<h2>{LABELS[name]}</h2>")
        config = configs[name]
        preparation = json.loads(
            resolve_path(config, "processed_corpus")
            .with_suffix(".json")
            .read_text(encoding="utf-8")
        )
        largest = next(s for s in preparation["subsets"] if s["budget"] == max(budgets))
        shares = ", ".join(
            f"{domain}: {100 * count / largest['actual_tokens']:.2f}%"
            for domain, count in largest["domain_tokens"].items()
        )
        blocks.append(f"<p>10M subset word shares: {escape(shares)}.</p>")
        values = []
        for budget in budgets:
            for order in orders:
                r = index[name, budget, order]
                values.append(
                    [
                        labels[budget],
                        f"{r['actual_tokens']:,}",
                        str(order),
                        f"{100 * r['exact_match_accuracy']:.1f}",
                        f"{100 * r['wer']:.2f}",
                        f"{100 * r['cer']:.2f}",
                        f"{r['mrr']:.4f}",
                        f"{100 * r['oov_references']:.2f}",
                        f"{r['training_seconds']:.3f}",
                        f"{r['binary_bytes'] / 2**20:.2f}",
                        f"{r['milliseconds_per_record']:.4f}",
                        f"{100 * r['random_top_tie_accuracy']:.2f}",
                    ]
                )
        add_table(
            [
                "Budget",
                "Actual words",
                "Order",
                "Accuracy %",
                "WER %",
                "CER %",
                "MRR",
                "Ref OOV %",
                "Train+build s",
                "Binary MiB",
                "Score ms/item",
                "Random top-tie %",
            ],
            values,
        )

    blocks.append("<h2>All models: exact-match accuracy (%)</h2>")
    add_table(
        ["Budget", "Order", *[LABELS[name] for name in configs]],
        [
            [
                labels[budget],
                str(order),
                *[
                    f"{100 * index[name, budget, order]['exact_match_accuracy']:.1f}"
                    for name in configs
                ],
            ]
            for budget in budgets
            for order in orders
        ],
    )
    blocks.append("<h2>Best model from each corpus</h2>")
    best_rows = {}
    for name in configs:
        candidates = [r for r in rows if r["corpus"] == name]
        best_rows[name] = max(
            candidates, key=lambda r: (r["exact_match_accuracy"], -r["wer"], -r["binary_bytes"])
        )
    add_table(
        [
            "Corpus",
            "Budget",
            "Order",
            "Correct / 500",
            "Accuracy %",
            "WER %",
            "Binary MiB",
            "Random top-tie %",
        ],
        [
            [
                LABELS[name],
                labels[r["budget"]],
                str(r["order"]),
                str(round(500 * r["exact_match_accuracy"])),
                f"{100 * r['exact_match_accuracy']:.1f}",
                f"{100 * r['wer']:.2f}",
                f"{r['binary_bytes'] / 2**20:.2f}",
                f"{100 * r['random_top_tie_accuracy']:.2f}",
            ]
            for name, r in best_rows.items()
        ],
    )
    news_best = best_rows["news_only"]
    reference_best = best_rows["all_five"]
    news_wins = sum(
        index["news_only", budget, order]["exact_match_accuracy"]
        > max(
            index[name, budget, order]["exact_match_accuracy"]
            for name in configs
            if name != "news_only"
        )
        for budget in budgets
        for order in orders
    )
    news_bigram = index["news_only", 10000000, 2]
    news_trigram = index["news_only", 10000000, 3]
    delta_points = 100 * (
        news_best["exact_match_accuracy"] - reference_best["exact_match_accuracy"]
    )
    blocks.extend(
        [
            "<h2>Observed findings</h2>",
            f"<p>News-only has the highest accuracy in {news_wins}/9 matched size/order "
            f"comparisons. Its best accuracy is {100 * news_best['exact_match_accuracy']:.1f}%, "
            f"{delta_points:.1f} "
            "percentage points above the original five-domain best. This is consistent with "
            "news training text matching the evaluation titles more closely.</p>",
            f"<p>At 10M words, the news bigram scores "
            f"{100 * news_bigram['exact_match_accuracy']:.1f}% and the news trigram scores "
            f"{100 * news_trigram['exact_match_accuracy']:.1f}%. Their WER values are "
            f"{100 * news_bigram['wer']:.2f}% and {100 * news_trigram['wer']:.2f}%, and binary "
            f"sizes are {news_bigram['binary_bytes'] / 2**20:.2f} and "
            f"{news_trigram['binary_bytes'] / 2**20:.2f} MiB, respectively. The bigram offers "
            "a favorable storage/accuracy trade-off in this run.</p>",
            f"<p>At 10M words, the news+wiki corpus has reference OOV "
            f"{100 * index['news_wiki', 10000000, 3]['oov_references']:.2f}%, versus "
            f"{100 * news_trigram['oov_references']:.2f}% for news alone, while its trigram "
            f"accuracy is {100 * index['news_wiki', 10000000, 3]['exact_match_accuracy']:.1f}%. "
            "Improved vocabulary coverage alone therefore does not guarantee better selection.</p>",
        ]
    )
    blocks.extend(
        [
            "<p>Comparisons are descriptive: the best models are selected on this same "
            "500-record set. One seed and one timing pass were used. Corpus size and "
            "order are controlled, but source vocabulary and phrase distributions differ. "
            "Exact-sentence leakage checks do not exclude all partial or semantic overlap. "
            "Model bytes are disk size, not measured RAM. Single-run latency includes "
            "normalization/scoring/ranking and excludes loading, metrics and output I/O.</p>",
            f"<p>All numerical results: results/metrics/{escape(prefix)}_domain_comparison.csv. "
            "Per-model metrics and full candidate rankings remain saved under their run IDs. "
            "Configurations: configs/news_only.yaml, configs/wiki_only.yaml, "
            "configs/news_wiki.yaml.</p>",
        ]
    )
    target = PROJECT_ROOT / "report" / f"{prefix}_domain_comparison.html"
    target.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Domain comparison</title>'
        '<style>body{font:16px/1.6 "Segoe UI",sans-serif;color:#183047;max-width:1400px;'
        "margin:36px auto;padding:0 24px}h1,h2{color:#173957}.scroll{overflow-x:auto}"
        "table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}"
        "th,td{text-align:left;padding:8px;border-bottom:1px solid #d8e2ea;white-space:nowrap}"
        "th{background:#e8eff6}tbody tr:nth-child(even){background:#f7f9fb}</style><body>"
        + "".join(blocks)
        + "</body></html>",
        encoding="utf-8",
    )
    assert file_sha256(original_path) == original_hash, "Original comparison was modified"
    write_json(
        PROJECT_ROOT / "report" / f"{prefix}_domain_comparison.json",
        {
            "runs": run_ids,
            "models_compared": len(rows),
            "reference_summary_sha256": original_hash,
            "csv_sha256": file_sha256(output_csv),
            "html_sha256": file_sha256(target),
            "best_models": best_rows,
            "original_run_preserved": True,
        },
    )
    print(f"Saved comparison of {len(rows)} models: {target}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="20261124")
    parser.add_argument("--reference-run", default="20261124_main")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    reference = PROJECT_ROOT / "results/metrics" / f"{args.reference_run}_summary.csv"
    before = file_sha256(reference)
    if not args.compare_only:
        if args.prepare:
            # Scan each source only once; the single-domain preparations reuse these caches.
            mixed = load_config(PROJECT_ROOT / "configs/news_wiki.yaml")
            configure_logging(mixed, f"{args.prefix}_domain_preparation")
            prepare_corpus(mixed, workers=2)
            for name in ("news_only", "wiki_only"):
                prepare_corpus(load_config(PROJECT_ROOT / "configs" / f"{name}.yaml"), workers=1)
        for name in STUDIES:
            config = load_config(PROJECT_ROOT / "configs" / f"{name}.yaml")
            run_experiments(config, f"{args.prefix}_{name}")
    assert file_sha256(reference) == before, "Original run must remain unchanged"
    export_comparison(args.prefix, args.reference_run)
