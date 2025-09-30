# tools/logger.py
import csv
from pathlib import Path

class CSVLogger:
    def __init__(self, path: str = "reports/metrics.csv"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._header_written = self.path.exists()

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  test_loss: float, test_acc: float):
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            if not self._header_written:
                w.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc"])
                self._header_written = True
            w.writerow([epoch,
                        f"{train_loss:.6f}",
                        f"{train_acc:.6f}",
                        f"{test_loss:.6f}",
                        f"{test_acc:.6f}"])

    def log_summary(self, tag: str, value: str):
        """Append a key/value pair at the end for convenience."""
        with self.path.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([tag, value])
