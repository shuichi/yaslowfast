from __future__ import annotations

import csv
import io
import json
from html.parser import HTMLParser

import pytest

from yaslowfast.metrics import VideoClassificationAccumulator
from yaslowfast.report import format_console_report, write_html_report


def evaluation_report(mapping=None):
    mapping = mapping or {"positive": 0, "negative": 1}
    accumulator = VideoClassificationAccumulator()
    # Distinct cell counts catch transpose and class-order mistakes: TP=3, FN=1, FP=2, TN=4.
    cases = [("positive", "positive")] * 3 + [("positive", "negative")]
    cases += [("negative", "positive")] * 2 + [("negative", "negative")] * 4
    for index, (target, prediction) in enumerate(cases):
        probabilities = [0.1, 0.1]
        probabilities[mapping[prediction]] = 0.9
        accumulator.update(
            [probabilities, probabilities],
            [mapping[target]] * 2,
            [str(index)] * 2,
            [f"{index}-0", f"{index}-1"],
            loss_sum=1.0,
        )
    report = accumulator.compute()
    report.update(
        class_to_idx=mapping,
        checkpoint="/models/best.pt",
        inputs={"positive": "/data/positive.tar", "negative": "/data/negative.tar"},
        mode="archives",
        evaluated_at="2026-10-30T08:00:00+00:00",
        preprocessing={"fps": 15, "window_frames": 64},
        normalization={"mean": [0.4] * 3, "std": [0.2] * 3},
        elapsed_seconds=12.5,
    )
    for row in report["predictions"]:
        row.update(
            source_archive="/data/動画.tar",
            source_member=f"nested/video_{row['video_id']}.mp4",
            sha256=f"digest-{row['video_id']}",
        )
    return report


class TableParser(HTMLParser):
    """Extract the rendered matrix text without depending on styling or whitespace."""

    def __init__(self):
        super().__init__()
        self.in_matrix = False
        self.in_cell = False
        self.rows = []
        self.current_row = []
        self.current_cell = ""

    def handle_starttag(self, tag, attrs):
        if tag == "table" and dict(attrs).get("class") == "matrix":
            self.in_matrix = True
        if not self.in_matrix:
            return
        if tag == "tr":
            self.current_row = []
        if tag in {"th", "td"}:
            self.in_cell = True
            self.current_cell = ""

    def handle_endtag(self, tag):
        if not self.in_matrix:
            return
        if tag in {"th", "td"}:
            self.current_row.append(self.current_cell)
            self.in_cell = False
        elif tag == "tr":
            self.rows.append(self.current_row)
        elif tag == "table":
            self.in_matrix = False

    def handle_data(self, data):
        if self.in_cell:
            self.current_cell += data


@pytest.mark.parametrize(
    "mapping", [{"positive": 0, "negative": 1}, {"positive": 1, "negative": 0}]
)
def test_console_and_html_keep_positive_semantics_for_both_id_orders(tmp_path, mapping):
    report = evaluation_report(mapping)
    console = format_console_report(report)
    assert "TP=3  FN=1  FP=2  TN=4" in console
    assert "Accuracy: 0.7000" in console
    assert "Balanced Accuracy: 0.7083" in console
    assert "Specificity (negative Recall): 0.6667" in console
    assert "評価動画: 10本 / 時間窓: 20" in console
    assert "行: 正解 / 列: 予測" in console

    output = write_html_report(report, tmp_path / "report")
    parser = TableParser()
    parser.feed(output.read_text())
    assert parser.rows == [
        ["正解 ＼ 予測", "予測 positive", "予測 negative", "合計"],
        ["positive", "3TP · 75.0%", "1FN · 25.0%", "4"],
        ["negative", "2FP · 33.3%", "4TN · 66.7%", "6"],
    ]


def test_html_escapes_untrusted_sources_and_metadata(tmp_path):
    report = evaluation_report()
    hostile = '<script>alert("name")</script> & sample.mp4'
    report["predictions"][0]["source_member"] = hostile
    report["checkpoint"] = "</pre><img src=x onerror=alert(1)>"
    output = write_html_report(report, tmp_path)
    rendered = output.read_text()
    assert '<script>alert("name")</script>' not in rendered
    assert "&lt;script&gt;alert(&quot;name&quot;)&lt;/script&gt; &amp; sample.mp4" in rendered
    assert "<img src=x" not in rendered
    assert "&lt;/pre&gt;&lt;img src=x onerror=alert(1)&gt;" in rendered
    # The only script is our static filter; no user strings are embedded in it.
    assert rendered.count("<script>") == 1
    assert "<script src=" not in rendered
    assert '<link rel="stylesheet"' not in rendered


def test_report_exports_original_data_and_preserves_other_files(tmp_path):
    report = evaluation_report({"positive": 1, "negative": 0})
    report["predictions"][0]["source_member"] = '動画/引用,"改行\nあり.mp4'
    (tmp_path / "keep.txt").write_text("unrelated")
    (tmp_path / "index.html").write_text("old report")
    index = write_html_report(report, tmp_path)
    assert index == tmp_path / "index.html"
    assert (tmp_path / "keep.txt").read_text() == "unrelated"
    assert json.loads((tmp_path / "metrics.json").read_text()) == report
    rows = list(csv.DictReader(io.StringIO((tmp_path / "predictions.csv").read_text())))
    assert len(rows) == 10
    assert rows[0]["source_member"] == report["predictions"][0]["source_member"]
    assert rows[0]["target_label"] == rows[0]["predicted_label"] == "positive"
    assert rows[0]["target"] == "1"
    assert rows[0]["result"] == "TP"
    assert float(rows[0]["probability_positive"]) == pytest.approx(0.9)
    assert float(rows[0]["probability_negative"]) == pytest.approx(0.1)
    assert rows[0]["sha256"] == "digest-0"
    assert rows[0]["num_windows"] == "2"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "index.html",
        "keep.txt",
        "metrics.json",
        "predictions.csv",
    ]


def test_generic_legacy_labels_do_not_invent_positive_class(tmp_path):
    report = evaluation_report()
    report["class_to_idx"] = {"<red>": 0, "blue": 1}
    rendered = write_html_report(report, tmp_path).read_text()
    console = format_console_report(report)
    assert "TP=" not in console
    assert "Specificity" not in console
    assert "予測 &lt;red&gt;" in rendered
    assert 'value="FP"' not in rendered
    assert 'data-result="誤分類"' in rendered
    rows = list(csv.DictReader(io.StringIO((tmp_path / "predictions.csv").read_text())))
    assert rows[0]["target_label"] == "<red>"
    assert rows[0]["probability_positive"] == ""


def test_prepared_dataset_rows_use_source_or_video_id(tmp_path):
    report = evaluation_report()
    for row in report["predictions"]:
        row.pop("source_archive")
        row.pop("source_member")
    report["predictions"][0]["source"] = "/videos/original.mp4"
    rendered = write_html_report(report, tmp_path).read_text()
    assert "/videos/original.mp4" in rendered
    rows = list(csv.DictReader(io.StringIO((tmp_path / "predictions.csv").read_text())))
    assert rows[0]["source"] == "/videos/original.mp4"
    assert rows[1]["source"] == "1"


def test_absent_class_metrics_have_no_nan_or_division_error(tmp_path):
    accumulator = VideoClassificationAccumulator()
    accumulator.update([[0.9, 0.1]], [0], ["a"], ["a0"], loss_sum=0.1)
    report = accumulator.compute()
    report["class_to_idx"] = {"positive": 0, "negative": 1}
    console = format_console_report(report)
    assert "Specificity (negative Recall): 0.0000" in console
    rendered = write_html_report(report, tmp_path).read_text()
    parser = TableParser()
    parser.feed(rendered)
    assert parser.rows[2] == ["negative", "0FP · 0.0%", "0TN · 0.0%", "0"]
    assert "分母が0の指標は0" in rendered
    assert "同率の場合はクラスID 0" in rendered


def test_invalid_report_does_not_overwrite_existing_outputs(tmp_path):
    report = evaluation_report()
    index = write_html_report(report, tmp_path)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    report["accuracy"] = float("nan")
    with pytest.raises(ValueError):
        write_html_report(report, tmp_path)
    assert index.exists()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_html_shows_evaluation_transforms_and_preprocessing_mismatch(tmp_path):
    report = evaluation_report()
    report.update(
        transforms={"crop_size": 224, "num_frames": 32, "alpha": 4, "eval_short_side": 256},
        batch_size=8,
        preprocessing_mismatch={"fps": {"checkpoint": 15, "dataset": 20}},
    )
    rendered = write_html_report(report, tmp_path).read_text()
    assert "crop_size&quot;: 224" in rendered
    assert "batch_size&quot;: 8" in rendered
    assert "前処理条件の差異" in rendered
    assert "異なる前処理条件を明示的に許可" in rendered
    assert "dataset&quot;: 20" in rendered


def test_atomic_write_failure_preserves_previous_html_and_cleans_temporary(tmp_path, monkeypatch):
    import yaslowfast.report as reporting

    index = tmp_path / "index.html"
    index.write_text("previous report")
    original_replace = reporting.os.replace

    def fail_html(source, destination):
        if destination.name == "index.html":
            raise OSError("disk write failed")
        original_replace(source, destination)

    monkeypatch.setattr(reporting.os, "replace", fail_html)
    with pytest.raises(OSError, match="disk write failed"):
        write_html_report(evaluation_report(), tmp_path)
    assert index.read_text() == "previous report"
    assert not list(tmp_path.glob(".*.tmp"))
