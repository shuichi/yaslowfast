"""Readable console and self-contained HTML reports for video evaluation."""

from __future__ import annotations

import csv
import html
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def _classes(report: dict[str, Any]) -> list[tuple[str, int]]:
    mapping = report["class_to_idx"]
    if set(mapping) == {"positive", "negative"}:
        return [(name, mapping[name]) for name in ("positive", "negative")]
    return sorted(mapping.items(), key=lambda item: item[1])


def _binary_counts(report: dict[str, Any]) -> dict[str, int] | None:
    mapping = report["class_to_idx"]
    if set(mapping) != {"positive", "negative"}:
        return None
    positive, negative = mapping["positive"], mapping["negative"]
    matrix = report["confusion_matrix"]
    return {
        "TP": matrix[positive][positive],
        "FN": matrix[positive][negative],
        "FP": matrix[negative][positive],
        "TN": matrix[negative][negative],
    }


def _result(row: dict[str, Any], mapping: dict[str, int]) -> str:
    if set(mapping) == {"positive", "negative"}:
        if row["target"] == mapping["positive"]:
            return "TP" if row["prediction"] == row["target"] else "FN"
        return "TN" if row["prediction"] == row["target"] else "FP"
    return "正解" if row["prediction"] == row["target"] else "誤分類"


def _source(row: dict[str, Any]) -> str:
    if row.get("source_archive") and row.get("source_member"):
        return f"{row['source_archive']}::{row['source_member']}"
    return str(row.get("source") or row["video_id"])


def format_console_report(report: dict[str, Any]) -> str:
    """Format metrics, preserving model class IDs but displaying positive first."""
    classes = _classes(report)
    matrix = report["confusion_matrix"]
    width = max(12, *(len(name) + 2 for name, _ in classes))
    lines = [
        "",
        "動画分類の評価結果",
        f"評価動画: {report['num_videos']}本 / 時間窓: {report['num_windows']}",
        "Confusion Matrix（行: 正解 / 列: 予測、単位: 動画）",
        " " * width + "".join(f"{name:>{width}}" for name, _ in classes),
    ]
    lines.extend(
        f"{name:<{width}}"
        + "".join(f"{matrix[index][predicted]:>{width}d}" for _, predicted in classes)
        for name, index in classes
    )
    counts = _binary_counts(report)
    if counts is not None:
        lines.append("  ".join(f"{name}={value}" for name, value in counts.items()))
    lines.extend(
        [
            f"Accuracy: {report['accuracy']:.4f}",
            f"Balanced Accuracy: {report['balanced_accuracy']:.4f}",
            f"Macro F1: {report['macro_f1']:.4f}",
            "",
            f"{'Class':<{width}} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>9}",
        ]
    )
    per_class = {row["class_id"]: row for row in report["per_class"]}
    for name, index in classes:
        row = per_class[index]
        lines.append(
            f"{name:<{width}} {row['precision']:>10.4f} {row['recall']:>10.4f} "
            f"{row['f1']:>10.4f} {row['support']:>9d}"
        )
    if counts is not None:
        total = counts["TN"] + counts["FP"]
        specificity = counts["TN"] / total if total else 0.0
        lines.append(f"Specificity (negative Recall): {specificity:.4f}")
    lines.append("各動画の時間窓の確率を平均して判定。同率はクラスID 0。未定義の指標は0。")
    return "\n".join(lines)


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _prediction_rows(report: dict[str, Any]) -> str:
    classes = _classes(report)
    names = {index: name for name, index in classes}
    rows = []
    # Show errors first, most confident errors first. All videos remain present.
    predictions = sorted(
        report["predictions"],
        key=lambda row: (
            row["target"] == row["prediction"],
            -row["probabilities"][row["prediction"]],
            _source(row),
        ),
    )
    for row in predictions:
        wrong = row["target"] != row["prediction"]
        result = _result(row, report["class_to_idx"])
        source = _source(row)
        cells = [
            f'<td><span class="badge {"wrong" if wrong else "correct"}">'
            f"{_escape(result)}</span></td>",
            f'<td class="source">{_escape(source)}</td>',
            f"<td>{_escape(names[row['target']])}</td>",
            f"<td>{_escape(names[row['prediction']])}</td>",
        ]
        cells.extend(f"<td>{row['probabilities'][index]:.4f}</td>" for _, index in classes)
        cells.append(f"<td>{row['num_windows']}</td>")
        rows.append(
            f'<tr data-result="{_escape(result)}" data-error="{str(wrong).lower()}">'
            + "".join(cells)
            + "</tr>"
        )
    return "\n".join(rows)


def _html_report(report: dict[str, Any]) -> str:
    classes = _classes(report)
    matrix = report["confusion_matrix"]
    counts = _binary_counts(report)
    rows = []
    for name, index in classes:
        support = sum(matrix[index])
        cells = [f'<th scope="row">{_escape(name)}</th>']
        for predicted_name, predicted in classes:
            count = matrix[index][predicted]
            fraction = count / support if support else 0.0
            color = "17, 122, 101" if index == predicted else "183, 65, 52"
            tag = ""
            if counts is not None:
                tag = (
                    ("TP" if predicted_name == "positive" else "FN")
                    if name == "positive"
                    else ("FP" if predicted_name == "positive" else "TN")
                )
            cells.append(
                f'<td style="background:rgba({color},{0.06 + 0.20 * fraction:.3f})">'
                f"<strong>{count}</strong><span>{tag} · {fraction:.1%}</span></td>"
            )
        cells.append(f'<td class="support">{support}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    headers = "".join(f'<th scope="col">予測 {_escape(name)}</th>' for name, _ in classes)
    per_class = {row["class_id"]: row for row in report["per_class"]}
    class_rows = []
    for name, index in classes:
        row = per_class[index]
        class_rows.append(
            f'<tr><th scope="row">{_escape(name)}</th>'
            f"<td>{row['precision']:.4f}</td><td>{row['recall']:.4f}</td>"
            f"<td>{row['f1']:.4f}</td><td>{row['support']}</td></tr>"
        )
    error_count = sum(row["target"] != row["prediction"] for row in report["predictions"])
    binary_note = ""
    filter_options = '<option value="all">すべて</option><option value="errors">誤分類のみ</option>'
    if counts is not None:
        binary_note = (
            f"TP {counts['TP']} / FN {counts['FN']} / FP {counts['FP']} / TN {counts['TN']}"
            " — FN: positiveの見逃し、FP: negativeをpositiveと誤判定。"
        )
        filter_options += '<option value="FN">FNのみ</option><option value="FP">FPのみ</option>'
    metadata = {
        key: report[key]
        for key in (
            "evaluated_at",
            "checkpoint",
            "mode",
            "inputs",
            "dataset",
            "split",
            "device",
            "precision",
            "elapsed_seconds",
            "class_to_idx",
            "preprocessing",
            "transforms",
            "normalization",
            "batch_size",
            "preprocessing_mismatch",
        )
        if key in report
    }
    probabilities_header = "".join(f"<th>P({_escape(name)})</th>" for name, _ in classes)
    checkpoint = _escape(report.get("checkpoint", "未指定"))
    mismatch_note = ""
    if report.get("preprocessing_mismatch"):
        mismatch = _escape(
            json.dumps(report["preprocessing_mismatch"], ensure_ascii=False, indent=2)
        )
        mismatch_note = (
            '<section class="mismatch" aria-labelledby="mismatch-title">'
            '<h2 id="mismatch-title">前処理条件の差異</h2>'
            "<p>この評価では、チェックポイントと異なる前処理条件を明示的に許可しています。"
            "同じ前処理条件での結果とは区別して比較してください。</p>"
            f"<pre>{mismatch}</pre></section>"
        )
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>動画分類の評価レポート | yaslowfast</title>
<style>
:root {{ color-scheme: light; font-family: system-ui, -apple-system, sans-serif;
  color: #173142; background: #f2f5f7; line-height: 1.65; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
main {{ max-width: 1180px; margin: auto; padding: 40px 24px 64px; }}
h1 {{ margin: 0; font-size: clamp(1.6rem, 4vw, 2.2rem); letter-spacing: -.03em; }}
h2 {{ margin: 0 0 12px; font-size: 1.25rem; }}
p {{ margin: 8px 0 16px; }}
.eyebrow {{ color: #17665c; font-weight: 700; letter-spacing: .13em; font-size: .8rem; }}
.muted {{ color: #5c6c76; }}
.checkpoint, .source, pre {{ overflow-wrap: anywhere; word-break: break-word; }}
.checkpoint {{ margin: 12px 0 24px; font-size: .9rem; }}
.cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 24px 0; }}
.card, section {{ background: white; border: 1px solid #dce4e9; border-radius: 12px; }}
.card {{ padding: 20px; }}
.card span {{ display: block; color: #5c6c76; font-size: .85rem; }}
.card strong {{ display: block; font-size: 1.85rem; margin-top: 4px; }}
section {{ padding: 24px; margin-top: 20px; }}
.scroll {{ overflow-x: auto; }}
.scroll-hint {{ display: none; color: #5c6c76; font-size: .8rem; }}
table {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 12px 14px; text-align: right; border-bottom: 1px solid #e7edf0; }}
thead th {{ background: #f6f8fa; font-size: .85rem; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
.matrix {{ max-width: 760px; }}
.matrix td {{ text-align: center; border: 4px solid white; min-width: 100px; }}
.matrix td strong {{ display: block; font-size: 1.8rem; }}
.matrix td span {{ display: block; font-size: .85rem; }}
.matrix .support {{ color: #5c6c76; min-width: 65px; }}
.notes {{ color: #5c6c76; font-size: .85rem; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 16px 0; }}
select, input {{ font: inherit; border: 1px solid #a9b9c3; border-radius: 6px;
  padding: 7px 10px; background: white; max-width: 100%; }}
.predictions {{ font-size: .85rem; }}
.predictions .source {{ text-align: left; min-width: 240px; max-width: 440px; }}
.predictions td {{ vertical-align: top; }}
.badge {{ display: inline-block; font-weight: 700; border-radius: 4px; padding: 2px 8px; }}
.wrong {{ color: #923a2e; background: #fbe8e3; }}
.correct {{ color: #1b6c58; background: #e7f4ed; }}
.mismatch {{ border-color: #e3b86f; background: #fffbf2; }}
pre {{ white-space: pre-wrap; font-size: .85rem; background: #f6f8fa;
  padding: 16px; border-radius: 8px; }}
a {{ color: #17665c; text-underline-offset: 3px; }}
footer {{ margin-top: 24px; color: #5c6c76; font-size: .85rem; }}
[hidden] {{ display: none !important; }}
@media (max-width: 760px) {{ .cards {{ grid-template-columns: repeat(2, 1fr); }}
  main {{ padding: 24px 12px; }} section {{ padding: 16px; }}
  .scroll-hint {{ display: block; }} }}
@media print {{ body {{ background: white; }} main {{ padding: 0; }}
  section, .card {{ break-inside: avoid; }} .toolbar {{ display: none; }}
  .scroll {{ overflow: visible; }} .predictions .source {{ min-width: 0; }} }}
</style>
</head>
<body><main>
<div class="eyebrow">YASLOWFAST / EVALUATION</div>
<h1>動画分類の評価レポート</h1>
<p class="checkpoint muted">チェックポイント: {checkpoint}</p>
<div class="cards">
<div class="card"><span>評価動画数</span><strong>{report["num_videos"]}本</strong></div>
<div class="card"><span>Accuracy</span><strong>{report["accuracy"]:.1%}</strong></div>
<div class="card"><span>Balanced Accuracy</span><strong>{report["balanced_accuracy"]:.1%}</strong></div>
<div class="card"><span>Macro F1</span><strong>{report["macro_f1"]:.4f}</strong></div>
</div>
{mismatch_note}
<section aria-labelledby="matrix-title">
<h2 id="matrix-title">混同行列 / Confusion Matrix</h2>
<p class="muted">行は正解、列は予測。動画1本を1件として集計しています。</p>
<p class="scroll-hint">表は左右にスクロールできます。</p>
<div class="scroll" tabindex="0" role="region" aria-label="混同行列"><table class="matrix">
<thead><tr><th scope="col">正解 ＼ 予測</th>{headers}<th scope="col">合計</th></tr></thead>
<tbody>{"".join(rows)}</tbody></table></div>
<p class="notes">各セルは件数と、正解クラス内での割合です。緑は正解、赤は誤分類です。</p>
<p>{binary_note}</p>
</section>
<section aria-labelledby="metrics-title">
<h2 id="metrics-title">クラス別の指標</h2>
<p class="scroll-hint">表は左右にスクロールできます。</p>
<div class="scroll" tabindex="0" role="region" aria-label="クラス別の指標">
<table><thead><tr><th>クラス</th><th>Precision</th><th>Recall</th>
<th>F1</th><th>動画数</th></tr></thead><tbody>{"".join(class_rows)}</tbody></table></div>
<p class="notes">Precisionは予測したクラスの正解率、Recallは正解クラスを拾えた割合、
F1は両者の調和平均です。分母が0の指標は0とします。
Balanced Accuracyはデータが存在するクラスのRecallの平均、Macro F1は全クラスのF1の平均です。</p>
</section>
<section aria-labelledby="predictions-title">
<h2 id="predictions-title">動画別の結果 — 誤分類 {error_count}本</h2>
<p class="muted">誤分類を先頭に表示しています。出典は入力tarとtar内のパスで識別します。</p>
<div class="toolbar">
<label>表示 <select id="result-filter">{filter_options}</select></label>
<label>動画を検索 <input id="source-filter" type="search" placeholder="ファイル名・パス"></label>
<span id="visible-count" role="status">{report["num_videos"]}本を表示</span>
</div>
<p class="scroll-hint">表は左右にスクロールできます。</p>
<div class="scroll" tabindex="0" role="region" aria-label="動画別の結果">
<table class="predictions" id="predictions-table">
<thead><tr><th>結果</th><th>出典 / 動画</th><th>正解</th><th>予測</th>
{probabilities_header}<th>時間窓数</th></tr></thead>
<tbody>{_prediction_rows(report)}</tbody></table></div>
<p class="notes">Pは各クラスの予測確率です。合計{report["num_windows"]}個の時間窓を処理し、
動画ごとに確率を平均して最大のクラスを選びます。同率の場合はクラスID 0を選びます。</p>
</section>
<section aria-labelledby="metadata-title">
<h2 id="metadata-title">評価条件と再利用用データ</h2>
<p><a href="metrics.json" download>評価結果JSON</a> ·
<a href="predictions.csv" download>動画別の結果CSV</a></p>
<pre>{_escape(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False))}</pre>
</section>
<footer>yaslowfast — このHTMLは外部サービスやネットワーク接続なしで閲覧できます。</footer>
</main>
<script>
const resultFilter = document.getElementById('result-filter');
const sourceFilter = document.getElementById('source-filter');
const predictionRows = document.querySelectorAll('#predictions-table tbody tr');
function filterRows() {{
  const selected = resultFilter.value;
  const query = sourceFilter.value.toLocaleLowerCase();
  let visible = 0;
  for (const row of predictionRows) {{
    const matchesResult = selected === 'all' ||
      (selected === 'errors' ? row.dataset.error === 'true' : row.dataset.result === selected);
    const matchesSource = row.querySelector('.source').textContent.toLocaleLowerCase().includes(query);
    row.hidden = !(matchesResult && matchesSource);
    if (!row.hidden) visible += 1;
  }}
  document.getElementById('visible-count').textContent = `${{visible}}本を表示`;
}}
resultFilter.addEventListener('change', filterRows);
sourceFilter.addEventListener('input', filterRows);
</script>
</body></html>
"""


def _csv_report(report: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    fields = [
        "video_id",
        "source_archive",
        "source_member",
        "source",
        "sha256",
        "target",
        "target_label",
        "prediction",
        "predicted_label",
        "result",
        "probability_positive",
        "probability_negative",
        "probabilities",
        "num_windows",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    mapping = report["class_to_idx"]
    names = {index: name for name, index in mapping.items()}
    for row in report["predictions"]:
        output = {key: row.get(key, "") for key in fields}
        output.update(
            source=_source(row),
            target_label=names[row["target"]],
            predicted_label=names[row["prediction"]],
            result=_result(row, mapping),
            probabilities=json.dumps(row["probabilities"], allow_nan=False),
        )
        for name in ("positive", "negative"):
            if name in mapping:
                output[f"probability_{name}"] = row["probabilities"][mapping[name]]
        writer.writerow(output)
    return stream.getvalue()


def _atomic_text(path: Path, contents: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            encoding="utf-8",
            newline="",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_html_report(report: dict[str, Any], directory: str | Path) -> Path:
    """Write an offline HTML report and its JSON/CSV data; return index.html.

    Each file is replaced atomically. Other files in the directory are preserved.
    Render everything before writing, so invalid data cannot publish a partial report.
    """
    documents = {
        "metrics.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        "predictions.csv": _csv_report(report),
        "index.html": _html_report(report),
    }
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, contents in documents.items():
        _atomic_text(directory / name, contents)
    return directory / "index.html"
