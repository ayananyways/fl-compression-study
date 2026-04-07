import csv
import os

import pytest

from src.training.metrics import MetricsTracker
from src.utils.logger import ResultsLogger, _CSV_COLUMNS


class TestResultsLogger:

    def test_creates_file_with_header(self, tmp_path: pytest.fixture) -> None:
        output_path = str(tmp_path / "results" / "test.csv")
        ResultsLogger(output_path=output_path, experiment_id="test_exp")
        assert os.path.exists(output_path)
        with open(output_path) as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == _CSV_COLUMNS

    def test_log_appends_row(self, tmp_path: pytest.fixture) -> None:
        output_path = str(tmp_path / "test.csv")
        log = ResultsLogger(output_path=output_path, experiment_id="exp1")
        metrics = MetricsTracker(round_num=1, train_loss=0.5, val_accuracy=0.75)
        log.log(metrics)

        with open(output_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["experiment_id"] == "exp1"
        assert rows[0]["round"] == "1"

    def test_multiple_rows(self, tmp_path: pytest.fixture) -> None:
        output_path = str(tmp_path / "test.csv")
        log = ResultsLogger(output_path=output_path, experiment_id="exp2")
        for i in range(5):
            log.log(MetricsTracker(round_num=i))
        with open(output_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5

    def test_file_readable_immediately_after_log(self, tmp_path: pytest.fixture) -> None:
        output_path = str(tmp_path / "test.csv")
        log = ResultsLogger(output_path=output_path, experiment_id="flush_test")
        log.log(MetricsTracker(round_num=0, train_loss=0.99))
        with open(output_path) as f:
            content = f.read()
        assert "0.99" in content

    def test_does_not_overwrite_existing(self, tmp_path: pytest.fixture) -> None:
        output_path = str(tmp_path / "test.csv")
        log1 = ResultsLogger(output_path=output_path, experiment_id="e1")
        log1.log(MetricsTracker(round_num=1))
        log2 = ResultsLogger(output_path=output_path, experiment_id="e1")
        log2.log(MetricsTracker(round_num=2))
        with open(output_path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
