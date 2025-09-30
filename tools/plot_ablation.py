# tools/plot_ablation.py
# CSV format example:
# experiment,param_name,param_value,test_acc
# K_sweep,K,1,0.9789
# K_sweep,K,2,0.9751
# ...

import csv, sys
import matplotlib.pyplot as plt

def load_ablation(path):
    xs, ys = [], []
    label = None
    with open(path, newline='') as f:
        rd = csv.reader(f)
        header = next(rd, None)
        for row in rd:
            exp, pname, pval, acc = row[0], row[1], row[2], row[3]
            label = f"{exp} ({pname})"
            try:
                x = float(pval)
            except Exception:
                x = pval
            y = float(acc)
            xs.append(x); ys.append(y)
    return label, xs, ys

def main(path, out_path):
    label, xs, ys = load_ablation(path)
    # sort if numeric
    try:
        xsf = list(map(float, xs))
        idx = sorted(range(len(xsf)), key=lambda i: xsf[i])
        xs = [xs[i] for i in idx]
        ys = [ys[i] for i in idx]
    except Exception:
        pass

    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.xlabel("parameter value")
    plt.ylabel("test accuracy")
    if label: plt.title(label)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

if __name__ == "__main__":
    in_csv = sys.argv[1] if len(sys.argv) > 1 else "reports/ablation_bits.csv"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/ablation_bits.png"
    main(in_csv, out_png)
