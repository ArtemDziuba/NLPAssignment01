"""Build an audited seven-page PDF and standalone HTML report from one experiment run."""

from __future__ import annotations

import argparse
import base64
import json
import re
from datetime import datetime
from html import escape
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from asr_rescoring.analysis import audit_run  # noqa: E402
from asr_rescoring.config import file_sha256, load_config, resolve_path, write_json  # noqa: E402


def build_report(run_id: str) -> Path:
    """Use saved measurements only; keep all derived diagnostics separately labeled."""
    if run_id != "20261124_main":
        raise ValueError("This narrative is authored for 20261124_main; review it for other runs")
    config = load_config()
    root = config["_root"]
    metrics_dir = resolve_path(config, "metrics")
    audit = audit_run(config, run_id)
    results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in metrics_dir.glob(f"{run_id}_order_*.json")
    ]
    results.sort(key=lambda r: (r["budget"], r["order"]))
    by_key = {(r["order"], r["budget"]): r for r in results}
    expected_accuracy = {
        (1, 100000): 0.736,
        (2, 100000): 0.780,
        (3, 100000): 0.786,
        (1, 1000000): 0.760,
        (2, 1000000): 0.824,
        (3, 1000000): 0.834,
        (1, 10000000): 0.774,
        (2, 10000000): 0.894,
        (3, 10000000): 0.902,
    }
    if any(
        by_key[key]["metrics"]["exact_match_accuracy"] != expected
        for key, expected in expected_accuracy.items()
    ):
        raise ValueError("Results changed; review the authored narrative before rebuilding")
    best = next(r for r in results if r["model_id"] == audit["best_model"])
    baselines = audit["baselines"]
    preparation = json.loads(
        resolve_path(config, "processed_corpus").with_suffix(".json").read_text(encoding="utf-8")
    )
    vocab_path = metrics_dir / f"{run_id}_vocabulary.json"
    vocabulary = json.loads(vocab_path.read_text(encoding="utf-8")) if vocab_path.exists() else None
    predictions = {
        key: {
            row["utt_id"]: row
            for row in (
                json.loads(line)
                for line in (resolve_path(config, "predictions") / result["predictions_file"])
                .read_text(encoding="utf-8")
                .splitlines()
            )
        }
        for key, result in by_key.items()
    }
    figure_dir = resolve_path(config, "figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = root / "report"
    output.mkdir(parents=True, exist_ok=True)
    font_dir = Path(matplotlib.get_data_path()) / "fonts/ttf"
    for suffix, filename in (
        ("", "DejaVuSans.ttf"),
        ("-Bold", "DejaVuSans-Bold.ttf"),
        ("-Italic", "DejaVuSans-Oblique.ttf"),
        ("-BoldItalic", "DejaVuSans-BoldOblique.ttf"),
    ):
        pdfmetrics.registerFont(TTFont("DVS" + suffix, str(font_dir / filename)))
    pdfmetrics.registerFontFamily(
        "DVS", normal="DVS", bold="DVS-Bold", italic="DVS-Italic", boldItalic="DVS-BoldItalic"
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
        }
    )
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="DVS-Bold",
            fontSize=22,
            leading=28,
            textColor=colors.HexColor("#15324f"),
            spaceAfter=12,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="DVS-Bold",
            fontSize=15,
            leading=20,
            textColor=colors.HexColor("#15324f"),
            spaceAfter=10,
        ),
        "sub": ParagraphStyle(
            "sub", fontName="DVS-Bold", fontSize=10.5, leading=15, spaceBefore=8, spaceAfter=5
        ),
        "body": ParagraphStyle("body", fontName="DVS", fontSize=9.5, leading=14, spaceAfter=8),
        "small": ParagraphStyle("small", fontName="DVS", fontSize=8, leading=11, spaceAfter=6),
        "cell": ParagraphStyle("cell", fontName="DVS", fontSize=8, leading=11),
        "caption": ParagraphStyle(
            "caption",
            fontName="DVS",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#526272"),
            spaceAfter=10,
        ),
    }
    story = []
    html = []
    page_titles = []

    def paragraph(text: str, style: str = "body") -> None:
        story.append(Paragraph(text, styles[style]))
        html.append(f'<p class="{style}">{text}</p>')

    def heading(text: str, first: bool = False) -> None:
        if page_titles:
            story.append(PageBreak())
            html.append("<hr>")
        page_titles.append(text)
        paragraph(text, "title" if first else "heading")

    def table(headers: list[str], rows: list[list], widths: list[float] | None = None) -> None:
        matrix = [
            [Paragraph(f"<b>{escape(str(value))}</b>", styles["cell"]) for value in headers]
        ] + [[Paragraph(escape(str(value)), styles["cell"]) for value in row] for row in rows]
        item = Table(matrix, colWidths=widths, repeatRows=1, hAlign="LEFT")
        item.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6edf5")),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f6f8fa")],
                    ),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.6, colors.HexColor("#b5c4d5")),
                ]
            )
        )
        story.extend([item, Spacer(1, 8)])
        html.append(
            "<table><thead><tr>"
            + "".join(f"<th>{escape(str(v))}</th>" for v in headers)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>" + "".join(f"<td>{escape(str(v))}</td>" for v in row) + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )

    def figure(fig, filename: str, width: float = 475, height: float = 210) -> None:
        path = figure_dir / f"{run_id}_{filename}.png"
        fig.savefig(path, dpi=180, facecolor="white")
        fig.savefig(path.with_suffix(".pdf"), facecolor="white")
        plt.close(fig)
        story.append(Image(str(path), width=width, height=height, kind="proportional"))
        html.append(
            '<img alt="'
            + filename
            + '" src="data:image/png;base64,'
            + base64.b64encode(path.read_bytes()).decode()
            + '">'
        )

    def pct(value: float, precision: int = 1) -> str:
        return f"{100 * value:.{precision}f}%"

    def label(budget: int) -> str:
        return {100000: "100K", 1000000: "1M", 10000000: "10M"}[budget]

    def raw(text: str) -> str:
        return escape(text)

    def predicted(key: tuple[int, int], identifier: str) -> dict:
        return predictions[key][identifier]

    def score_for(row: dict, text: str) -> dict:
        return next(score for score in row["ranking"] if score["raw"] == text)

    best_key = best["order"], best["budget"]
    best_metrics = best["metrics"]
    big = by_key[2, 10000000]
    total_input = sum(
        m["counts"]["input_sentences"] for m in preparation["domain_manifests"].values()
    )
    total_words = sum(m["tokens"] for m in preparation["domain_manifests"].values())
    date = datetime.strptime(run_id[:8], "%Y%m%d").strftime("%d %B %Y")

    heading("Ukrainian ASR candidate rescoring", first=True)
    paragraph(f"Classical n-gram language models | {date}<br/>Run: {run_id}", "small")
    paragraph(
        f"<b>Result.</b> The best language model was a trigram trained on "
        f"{best['actual_tokens']:,} words: <b>{pct(best_metrics['exact_match_accuracy'])}</b> "
        f"exact-match accuracy ({audit['best_correct']}/500), WER "
        f"{pct(best_metrics['wer'], 2)}, CER {pct(best_metrics['cer'], 2)}, and MRR "
        f"{best_metrics['mrr']:.4f}. Increasing corpus size improved all three n-gram orders."
    )
    paragraph(
        "<b>Critical interpretation.</b> The reference occurs as candidate 0 in every record. "
        "Consequently, selecting the first candidate achieves 100% accuracy and zero WER/CER. "
        "The experiment demonstrates language-model ranking behavior on synthetic ASR-like "
        "candidates; it does not establish an improvement over the provided candidate order."
    )
    paragraph("Experimental design", "sub")
    table(
        ["Factor", "Controlled setting"],
        [
            ["Models", "Unigram, bigram, trigram × 100K, 1M, 10M budgets = 9 models"],
            ["Training data", "UberText 2.0: court, fiction, news, social, wikipedia [1, 2]"],
            ["Evaluation", "500 fixed Ukrainian records; 8 candidates each; 4,000 candidates"],
            ["Selection", "Minimum assignment-defined perplexity; first original index on ties"],
            ["Reproducibility", "Seed 42; nested subsets; source/model hashes; config snapshots"],
            ["Environment", "CPU; KenLM 0.3.0; Linux container on Windows/WSL2; Python 3.11.18"],
        ],
        [90, 385],
    )
    paragraph("Baselines and attainable accuracy", "sub")
    table(
        ["Method", "Accuracy", "WER", "MRR"],
        [
            [
                "First candidate",
                pct(baselines["first_candidate"]["exact_match_accuracy"]),
                pct(baselines["first_candidate"]["wer"], 2),
                f"{baselines['first_candidate']['mrr']:.4f}",
            ],
            [
                "Seeded uniform random",
                pct(baselines["seeded_random"]["exact_match_accuracy"]),
                pct(baselines["seeded_random"]["wer"], 2),
                f"{baselines['seeded_random']['mrr']:.4f}",
            ],
            ["Oracle candidate availability", "100.0%", "—", "—"],
            [
                "Best LM: trigram, 10M",
                pct(best_metrics["exact_match_accuracy"]),
                pct(best_metrics["wer"], 2),
                f"{best_metrics['mrr']:.4f}",
            ],
        ],
        [220, 85, 85, 85],
    )
    paragraph(
        f"There are {audit['matching_candidates']} normalized reference-equivalent candidates "
        f"among 4,000 candidates, including {audit['records_with_multiple_normalized_matches']} "
        "records with a second matching candidate. Thus uniform random selection has analytical "
        f"expected accuracy {pct(audit['uniform_random_expected_accuracy'], 3)}, not simply 12.5%. "
        "The reported random baseline is one fixed seeded realization."
    )

    heading("1. Corpus preparation and scoring")
    paragraph(
        f"Five official sentence-split bz2 exports ({5.229237113:.2f} GB compressed) were scanned "
        f"once: {total_input:,} nonempty sentence occurrences. Each domain retained up to 200,000 "
        f"unique normalized sentences using seeded hash-priority sampling. The retained domain "
        f"samples contain {total_words:,} words before cross-domain deduplication. Interleaving "
        f"and global deduplication produced {preparation['sentences']:,} sentences."
    )
    table(
        ["Domain", "Scanned sentences", "Retained words", "10M word share"],
        [
            [
                domain,
                f"{manifest['counts']['input_sentences']:,}",
                f"{manifest['tokens']:,}",
                pct(
                    preparation["subsets"][-1]["domain_tokens"][domain]
                    / preparation["subsets"][-1]["actual_tokens"]
                ),
            ]
            for domain, manifest in preparation["domain_manifests"].items()
        ],
        [85, 140, 125, 125],
    )
    table(
        ["Nominal budget", "Actual word tokens", "Sentences"],
        [
            [label(row["budget"]), f"{row['actual_tokens']:,}", f"{row['sentences']:,}"]
            for row in preparation["subsets"]
        ],
        [150, 175, 150],
    )
    paragraph(
        "Mixture weights are approximately equal by sentence count, not word count. Court "
        "sentences are longer and supply about 40% of the 10M words. Smaller subsets are strict "
        "whole-sentence prefixes; the final sentence is never cut or skipped to force an exact "
        "budget. All models of a given size use exactly the same text."
    )
    paragraph(
        "<b>Normalization.</b> Apply Unicode NFC, lowercase, canonical apostrophes, Unicode "
        "letter/number tokenization with internal apostrophes, and whitespace collapse. Other "
        "punctuation becomes separators. No spelling correction, transliteration or stemming "
        "is applied. Corpus-only cleaning rejects URLs/obvious markup, non-Cyrillic lines, and "
        "sentences outside 2–150 words. Segmentation uses the upstream sentenced export."
    )
    paragraph(
        "<b>Leakage audit.</b> Normalized evaluation references and hypotheses are exclusions "
        "only. Exact training overlaps are removed before sampling; the final subsets passed "
        "a zero-overlap audit. Related source documents, partial overlaps and paraphrases "
        "are not excluded by this exact-sentence check."
    )
    paragraph(
        "<b>Scoring.</b> L = log10 P(sentence + EOS | BOS); H = −L/N; "
        "perplexity = 10<super>H</super>. "
        "N counts normalized words and excludes boundary tokens, following the assignment. "
        "KenLM’s built-in perplexity instead includes EOS in N and is not used [4]. "
        "Candidates are ranked by H to avoid exponent overflow; exact ties use original order."
    )
    paragraph(
        "<b>Unigrams.</b> KenLM estimates order 1, but its generic ARPA loader requires a "
        "higher-order container. A container with zero observed bigrams and zero backoffs "
        "preserves the original unigram probabilities. KenLM context is reset per word/EOS; "
        "native smoke tests verify the resulting sums against the source unigram ARPA. "
        "Bigram/trigram models use modified Kneser–Ney estimation [3], without pruning or "
        "fallback discounts. The scoring convention is fixed for the entire grid."
    )

    heading("2. Accuracy and corpus-size effects")
    table(
        ["Words", "Order", "Correct / 500", "Accuracy", "WER", "CER", "MRR"],
        [
            [
                label(r["budget"]),
                r["order"],
                round(r["metrics"]["exact_match_accuracy"] * 500),
                pct(r["metrics"]["exact_match_accuracy"]),
                pct(r["metrics"]["wer"], 2),
                pct(r["metrics"]["cer"], 2),
                f"{r['metrics']['mrr']:.4f}",
            ]
            for r in results
        ],
        [55, 45, 85, 75, 70, 70, 75],
    )
    fig, ax = plt.subplots(figsize=(7, 2.65))
    for order, color in [(1, "#9b6a36"), (2, "#287b8e"), (3, "#28479c")]:
        series = [r for r in results if r["order"] == order]
        ax.plot(
            [r["actual_tokens"] for r in series],
            [100 * r["metrics"]["exact_match_accuracy"] for r in series],
            marker="o",
            label=f"{order}-gram",
            color=color,
        )
    ax.axhline(100, color="#777777", linestyle="--", linewidth=1, label="First candidate / oracle")
    ax.set(
        xscale="log",
        ylim=(68, 102),
        xlabel="Actual training word tokens",
        ylabel="Exact-match accuracy (%)",
    )
    ax.set_xticks([99987, 999999, 9999984], ["100K", "1M", "10M"])
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)
    figure(fig, "accuracy_by_size", height=185)
    paragraph(
        "Figure 1. All model curves improve with more data; the positional baseline remains "
        "perfect. Raw and normalized exact-match accuracies coincide for every LM in this run.",
        "caption",
    )
    paragraph(
        "From 100K to 10M words, accuracy increases by <b>3.8 percentage points</b> for unigrams, "
        "<b>11.4 points</b> for bigrams, and <b>11.6 points</b> for trigrams. At 10M, adding "
        "context from unigram to trigram yields 12.8 points, while trigram over bigram adds "
        "only 0.8 points (four records). More data has a larger effect than adding the third "
        "order after bigrams are available."
    )
    second_rank = audit["best_error_reference_ranks"].get(
        "2", audit["best_error_reference_ranks"].get(2, 0)
    )
    paragraph(
        "The best model leaves 49 exact-match errors, but its reference is still ranked "
        f"second in {second_rank} "
        "of them. Its low WER/CER reflects mostly small local changes in otherwise correct "
        "titles. Oracle accuracy is 100% for every configuration; failures are ranking errors, "
        "not missing-reference cases. The test set is reused to compare all nine models, so "
        "“best” is a descriptive selection rather than an independent generalization estimate."
    )

    heading("3. Efficiency, model size and vocabulary")
    table(
        ["Words", "Order", "Train + build (s)", "Binary MiB", "Load (ms)", "Score ms/item"],
        [
            [
                label(r["budget"]),
                r["order"],
                f"{r['training_seconds']:.3f}",
                f"{r['binary_bytes'] / 2**20:.2f}",
                f"{1000 * r['metrics']['model_load_seconds']:.1f}",
                f"{r['metrics']['milliseconds_per_record']:.4f}",
            ]
            for r in results
        ],
        [55, 45, 100, 85, 90, 100],
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.5))
    largest = [r for r in results if r["order"] == 3]
    axes[0].plot(
        [r["actual_tokens"] for r in largest],
        [100 * r["metrics"]["oov_references"] for r in largest],
        "o-",
        label="References",
    )
    axes[0].plot(
        [r["actual_tokens"] for r in largest],
        [100 * r["metrics"]["oov_hypotheses"] for r in largest],
        "s-",
        label="All candidates",
    )
    axes[0].set(xscale="log", xlabel="Training word tokens", ylabel="OOV tokens (%)")
    axes[0].legend(fontsize=8)
    for order in (1, 2, 3):
        series = [r for r in results if r["order"] == order]
        axes[1].plot(
            [r["binary_bytes"] / 2**20 for r in series],
            [100 * r["metrics"]["exact_match_accuracy"] for r in series],
            "o-",
            label=f"{order}-gram",
        )
    axes[1].set(xscale="log", xlabel="Binary model (MiB)", ylabel="Accuracy (%)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.tight_layout()
    figure(fig, "efficiency_and_oov", height=175)
    paragraph(
        "Figure 2. Vocabulary coverage depends on corpus size; the size/accuracy plot "
        "shows the cost of adding orders. Model bytes are serialized size, not measured RAM.",
        "caption",
    )
    paragraph(
        "Reference OOV drops from 28.08% (100K) to 8.97% (1M) to 2.32% (10M). Candidate OOV "
        "falls from 35.33% to 10.87%. At a fixed corpus size the vocabulary is shared across "
        "orders, so higher-order gains cannot be attributed to a larger vocabulary."
    )
    paragraph(
        f"All nine models required {audit['total_training_seconds']:.2f} seconds of recorded "
        "training/build time. The successful runner completed in approximately 60.5 seconds "
        "(including validation, hashes, loading, metrics and exports). The 10M trigram occupies "
        f"{best['binary_bytes'] / 2**20:.2f} MiB, versus {big['binary_bytes'] / 2**20:.2f} MiB for "
        "the bigram: about 2.60× the storage for four extra correct records. The bigram is "
        "therefore an attractive resource/accuracy compromise on this evaluation."
    )
    paragraph(
        "These are single-run timings with warmed scoring after one warmup, on a Linux "
        "container with a 2G KenLM sorting-memory setting. Scoring includes normalization "
        "and ranking; loading, metrics, OOV scans and export are excluded. Per-token Python "
        "calls add unigram overhead. No repeated latency benchmark, peak-RSS measurement, "
        "GPU comparison, or speed-significance claim is made. An initial native crash was "
        "resolved by using Linux-local scratch files and copying completed models to Windows; "
        "the failed attempt is excluded from the successful timing."
    )

    heading("4. Qualitative successes and failures")
    examples = [
        ("A. Context resolves a case-ending error", "ua_title_0005", "success_case"),
        ("B. Context distinguishes a real-word substitution", "ua_title_0246", "success_word"),
        ("C. Repetition can lower average perplexity", "ua_title_0024", "repetition"),
        ("D. Apostrophe fragmentation changes tokenization", "ua_title_0317", "apostrophe"),
        (
            "E. Correct vocabulary coverage does not guarantee a correct name",
            "ua_title_0049",
            "entity",
        ),
    ]
    for title, identifier, kind in examples:
        row = predicted(best_key, identifier)
        reference_score = next(s for s in row["ranking"] if s["index"] == 0)
        chosen = row["ranking"][0]
        paragraph(f"{title} <font color='#657484'>({identifier})</font>", "sub")
        paragraph(f"<b>Reference:</b> {raw(row['reference'])}", "small")
        if kind.startswith("success"):
            unigram = predicted((1, 10000000), identifier)
            rival = score_for(row, unigram["selected"])
            paragraph(
                f"<b>Unigram selected:</b> {raw(unigram['selected'])}<br/>"
                "<b>Trigram:</b> selects the reference; "
                f"H(ref)={reference_score['cross_entropy']:.4f}, "
                f"H(unigram choice)={rival['cross_entropy']:.4f}.",
                "small",
            )
            explanation = (
                "The unigram prefers «із зоні АТО»; the trigram favors «із зони АТО». "
                "The result is consistent with local context helping select the appropriate "
                "case ending rather than relying only on word frequency."
                if kind == "success_case"
                else "Both «дрон» and «трон» are real words. The trigram selects the intended "
                "sentence containing «запустив дрон», while the unigram selects «трон». "
                "A fluent individual word can still be a recognition error in context."
            )
        else:
            paragraph(
                f"<b>Trigram selected:</b> {raw(row['selected'])}<br/>"
                f"<b>Perplexity:</b> reference {reference_score['perplexity']:,.1f}; "
                f"selected {chosen['perplexity']:,.1f}.",
                "small",
            )
            explanation = {
                "repetition": "The duplicated «відомої» is preferred even though all words "
                "are in vocabulary. Optimizing average word cost can reward "
                "an extra relatively cheap word. This is a ranking-objective failure.",
                "apostrophe": "The correct internal apostrophe stays within a token, while "
                "«д ' Італія» becomes separate tokens. Their different N values and "
                "contexts change perplexity. The evaluation deliberately does not "
                "repair this recognition error during normalization.",
                "entity": "Both «Бетіс» and the wrong «Бетис» are in vocabulary, but the model "
                "prefers the wrong spelling. More coverage and local fluency do not "
                "provide a factual guarantee about a named entity.",
            }[kind]
        paragraph(explanation, "small")
    repetitions = audit["best_error_categories"]["adjacent_repetition_only"]
    paragraph(
        f"Across all 49 best-model errors, {repetitions} "
        "are pure adjacent repetitions and "
        f"{audit['best_error_categories']['apostrophe_fragmentation_only']} are pure apostrophe "
        "fragmentation; the remaining 17 are other or combined errors. These categories are "
        "deterministic token-pattern diagnostics, not a complete linguistic annotation.",
        "caption",
    )

    heading("5. Position bias, exact ties and uncertainty")
    paragraph(
        "<b>Candidate order is informative in this dataset.</b> Candidate 0 is always the "
        "reference. This explains the perfect first-candidate baseline and also benefits "
        "first-index tie breaking whenever a reference ties a different normalized sentence. "
        "The official results above preserve that configured convention throughout."
    )
    table(
        ["Words", "Order", "Recorded accuracy", "Expected random top-tie accuracy"],
        [
            [
                label(r["budget"]),
                r["order"],
                pct(r["metrics"]["exact_match_accuracy"]),
                pct(
                    audit["tie_sensitivity"][f"{r['order']}gram_{r['budget']}"][
                        "random_tie_accuracy"
                    ],
                    2,
                ),
            ]
            for r in results
        ],
        [65, 55, 145, 210],
    )
    sensitivity = audit["tie_sensitivity"][f"{best['order']}gram_{best['budget']}"]
    paragraph(
        "<b>Analytical sensitivity diagnostic, not a second scored experiment.</b> For each "
        "saved exact top-score tie, the expected correctness under uniform tie selection is "
        "the fraction of tied candidates matching the reference. No scores or official "
        f"predictions are changed. The best model has "
        f"{sensitivity['top_ties_between_distinct_normalized_texts']} top ties involving "
        "different normalized sentences. Its accuracy can range from "
        f"{pct(sensitivity['minimum_accuracy'])} to {pct(sensitivity['maximum_accuracy'])} "
        f"under different exact-tie choices; uniform top-tie selection gives expected "
        f"{pct(sensitivity['random_tie_accuracy'], 2)}."
    )
    paragraph(
        "The small-corpus scores are especially sensitive: the 100K unigram falls from "
        "73.6% recorded accuracy to about 46.18% expected under uniform top-tie selection. "
        "Mapping unseen variants to the same unknown token can produce equal scores. "
        "Duplicate normalized candidates by themselves are not an accuracy error; the "
        "audit counts ties between distinct normalized texts separately."
    )
    lo, hi = audit["best_accuracy_wilson_95"]
    pair = audit["best_vs_bigram_paired"]
    paragraph(
        f"<b>Uncertainty.</b> A descriptive 95% Wilson interval for the best model's "
        f"451/500 correctness is {pct(lo, 2)}–{pct(hi, 2)}. This interval conditions on this "
        "evaluation convention and does not adjust for selecting the best of nine models. "
        f"Against the 10M bigram, the trigram wins on {pair['wins']} records and loses on "
        f"{pair['losses']}; the two-sided exact McNemar p-value is "
        f"{pair['exact_mcnemar_p']:.4f}. Four extra correct items are insufficient to establish "
        "a reliable superiority claim from this exploratory comparison."
    )
    if vocabulary:
        groups = vocabulary["reference_oov_by_outcome"]
        paragraph(
            f"Reference OOV tokens are {pct(groups['correct']['rate'], 2)} among correct "
            f"selections and {pct(groups['incorrect']['rate'], 2)} among errors. This association "
            "supports examining vocabulary coverage in failure analysis, but does not "
            "establish causality: length, rare entities and domain also differ."
        )

    heading("6. Conclusions, limitations and reproducibility")
    paragraph(
        "<b>Answer to the assignment.</b> Higher-order context and additional text improve "
        "candidate ranking in this grid. Trigrams lead at every corpus size, and bigram/trigram "
        "models benefit much more from a 100-fold corpus increase than unigrams. The 10M "
        "trigram gives the highest recorded LM accuracy, while the 10M bigram offers nearly "
        "the same accuracy with substantially less storage. Both outperform the seeded random "
        "baseline and both underperform the perfect positional baseline."
    )
    paragraph(
        "<b>Where the method works.</b> Learned local co-occurrences help distinguish plausible "
        "case endings and real-word substitutions. <b>Where it fails.</b> Fluency scores do "
        "not encode the intended utterance, acoustic evidence, or factual name identity. "
        "Average perplexity can favor repetitions and fragmented tokens. OOV collapse and "
        "the deterministic gold-first tie rule further complicate interpretation."
    )
    paragraph("Limitations and next experiments", "sub")
    paragraph(
        "The 500 examples are ASR-like title hypotheses rather than an independently decoded "
        "speech test set. References are always available and always first. All nine settings "
        "were compared on the same evaluation set, with one preparation seed and one timing "
        "run. Exact leakage filtering does not remove partial or semantic overlap. Court text "
        "supplies about 40% of words; the fixed mixture is not a domain-ablation study. "
        "Model size is not peak memory use, and single-run sub-millisecond times should not "
        "be treated as production latency estimates."
    )
    paragraph(
        "A stronger follow-up would use an independent held-out set with randomized candidate "
        "positions, an explicitly fixed order-neutral tie policy, repeated seeds/timings, and "
        "a genuine ASR top-1 baseline. Repetition or word-insertion penalties and acoustic/LM "
        "combination should be tuned on separate development data. The optional modern LLM "
        "comparison was not run; no neural-model or memory-use results are claimed."
    )
    paragraph("Reproduce and inspect", "sub")
    paragraph(
        "The report is generated from saved artifacts, with all metrics checked against "
        f"{audit['predictions_audited']:,} predictions and "
        f"{audit['candidate_scores_audited']:,} candidate scores. Nine model artifact sets "
        "passed checksum checks. The original assignment and evaluation file remain unchanged. "
        "All 39 unit/native integration tests passed after the scratch-file fix.",
        "small",
    )
    paragraph(
        f"<b>Run identifier:</b> {run_id}<br/>"
        f"<b>Comparison table:</b> results/metrics/{run_id}_summary.csv<br/>"
        f"<b>Diagnostics:</b> results/metrics/{run_id}_analysis.json<br/>"
        "<b>Per-record outputs:</b> results/predictions/ (one JSONL per model)<br/>"
        "<b>Models:</b> models/order_N_tokens_B/ (ARPA, binary, manifest, training log)<br/>"
        f"<b>Native tests:</b> results/logs/experiment_tests.xml<br/>"
        f"<b>Report rebuild:</b> uv run --locked --group analysis --group report python "
        f"scripts/build_report.py --run-id {run_id}",
        "small",
    )
    paragraph(
        "<b>Training command inside the provided Linux environment:</b><br/>"
        f"python scripts/run_experiments.py --run-id {run_id}<br/>"
        "Use the Docker mount and PYTHONPATH instructions in docs/development.md. "
        "Matching models are reused only when input, preprocessing, trainer and toolchain "
        "identities plus artifact hashes agree. Dataset preparation is not repeated.",
        "small",
    )
    paragraph("References", "sub")
    paragraph(
        "[1] Chaplynskyi, D. (2023). Introducing UberText 2.0: A Corpus of Modern Ukrainian "
        "at Scale. UNLP, pp. 1–10. "
        '<link href="https://aclanthology.org/2023.unlp-1.1/" '
        'color="#245b91">ACL Anthology</link>.<br/>'
        "[2] Lang-uk. UberText 2.0 official sentence-split exports. "
        '<link href="https://lang.org.ua/en/ubertext/" color="#245b91">Download page</link>.<br/>'
        "[3] Heafield et al. (2013). Scalable Modified Kneser–Ney Language Model Estimation. "
        '<link href="https://kheafield.com/code/kenlm/estimation/" '
        'color="#245b91">KenLM estimation</link>.<br/>'
        "[4] KenLM Python score/perplexity implementation. "
        '<link href="https://github.com/kpu/kenlm/blob/master/python/kenlm.pyx" '
        'color="#245b91">Source</link>.<br/>'
        "[5] Official local assignment: readme.md. "
        "Experiment conventions: configs/experiments.yaml.",
        "small",
    )

    pdf_path = output / "experiment_results.pdf"
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=55,
        rightMargin=55,
        topMargin=45,
        bottomMargin=44,
        title="Ukrainian ASR candidate rescoring: experiment results",
        author="ASR Rescoring Project",
    )

    def footer(canvas, document) -> None:
        canvas.setStrokeColor(colors.HexColor("#d5dfe8"))
        canvas.line(55, 34, A4[0] - 55, 34)
        canvas.setFont("DVS", 7)
        canvas.setFillColor(colors.HexColor("#657484"))
        canvas.drawString(55, 22, f"ASR rescoring | {run_id} | measured results")
        canvas.drawRightString(A4[0] - 55, 22, str(document.page))

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    html_path = output / "experiment_results.html"
    html_path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>ASR experiment results</title>'
        "<style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:16px/1.6 "
        '"Segoe UI",sans-serif;color:#182b3e} .title{font-size:34px;font-weight:700}'
        ".heading{font-size:26px;font-weight:700}.sub{font-weight:700}.small,.caption{font-size:14px}"
        ".caption{color:#526272}table{border-collapse:collapse;width:100%;margin:18px 0}"
        "th,td{padding:9px;text-align:left;border-bottom:1px solid #dce4ed}th{background:#e6edf5}"
        "img{max-width:100%;display:block;margin:24px auto}hr{margin:50px 0;border:0;"
        "border-top:1px solid #dce4ed}</style><body>"
        + re.sub(
            r'<link href="([^"]+)" color="[^"]+">([^<]+)</link>',
            r'<a href="\1">\2</a>',
            "".join(html),
        )
        .replace("<super>", "<sup>")
        .replace("</super>", "</sup>")
        + "</body></html>",
        encoding="utf-8",
    )
    write_json(
        output / "report_manifest.json",
        {
            "run_id": run_id,
            "pdf": pdf_path.name,
            "html": html_path.name,
            "planned_sections": page_titles,
            "audited_predictions": audit["predictions_audited"],
            "audited_candidate_scores": audit["candidate_scores_audited"],
            "sha256": {
                "pdf": file_sha256(pdf_path),
                "html": file_sha256(html_path),
                "summary": file_sha256(metrics_dir / f"{run_id}_summary.csv"),
                "analysis": file_sha256(metrics_dir / f"{run_id}_analysis.json"),
                "evaluation": file_sha256(resolve_path(config, "evaluation")),
                "assignment": file_sha256(root / "readme.md"),
            },
        },
    )
    return pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    arguments = parser.parse_args()
    print(build_report(arguments.run_id))
