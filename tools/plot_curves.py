# tools/plot_curves.py
import csv
import sys
import matplotlib.pyplot as plt

def load_metrics(path):
    epochs, tr_loss, tr_acc, te_loss, te_acc = [], [], [], [], []
    with open(path, newline='') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            # skip non-epoch lines (e.g., summary)
            try:
                e = int(row[0])
            except Exception:
                continue
            epochs.append(e)
            tr_loss.append(float(row[1]))
            tr_acc.append(float(row[2]))
            te_loss.append(float(row[3]))
            te_acc.append(float(row[4]))
    return epochs, tr_loss, tr_acc, te_loss, te_acc

def main(path="reports/metrics.csv"):
    epochs, tr_loss, tr_acc, te_loss, te_acc = load_metrics(path)

    # Accuracy vs epoch
    plt.figure()
    plt.plot(epochs, tr_acc, label="train acc")
    plt.plot(epochs, te_acc, label="test acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("Accuracy vs Epoch")
    plt.tight_layout()
    plt.savefig("reports/figures/curve_accuracy.png", dpi=200)
    plt.close()

    # Loss vs epoch
    plt.figure()
    plt.plot(epochs, tr_loss, label="train loss")
    plt.plot(epochs, te_loss, label="test loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.title("Loss vs Epoch")
    plt.tight_layout()
    plt.savefig("reports/figures/curve_loss.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "reports/metrics.csv"
    main(path)
