# tools/sweep.py
import os, csv, subprocess, sys, time, yaml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # tools/
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)               # repo root

BASE_CONFIG = os.path.join(PROJECT_ROOT, "experiments", "A_coherence_noise.yaml")
PYTHON_EXE  = sys.executable or "python"  # 使用当前解释器
MAIN_MOD    = "src.main"                  # 入口

def run_one(config_dict, outdir):
    os.makedirs(outdir, exist_ok=True)
    # 写临时 config
    cfg_path = os.path.join(outdir, "config.yaml")
    with open(BASE_CONFIG, "r") as f:
        base = yaml.safe_load(f)
    # 合并覆盖
    def deep_update(dst, src):
        for k,v in src.items():
            if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                deep_update(dst[k], v)
            else:
                dst[k] = v
    deep_update(base, config_dict)
    with open(cfg_path, "w") as f:
        yaml.safe_dump(base, f)

    # 运行训练
    cmd = [PYTHON_EXE, "-m", MAIN_MOD, "--config", cfg_path, "--outdir", outdir]
    print(">>>", " ".join(cmd))
    ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if ret.returncode != 0:
        raise RuntimeError(f"Run failed: {outdir}")

    # 读取 best_test_acc
    metrics_csv = os.path.join(outdir, "metrics.csv")
    best = None
    if os.path.exists(metrics_csv):
        with open(metrics_csv, newline="") as f:
            rd = csv.reader(f)
            for row in rd:
                if len(row) >= 2 and row[0] == "best_test_acc":
                    best = float(row[1])
    if best is None:
        # fallback: 取 test_acc 最大值
        with open(metrics_csv, newline="") as f:
            rd = csv.reader(f)
            header = next(rd, None)
            best = 0.0
            for row in rd:
                try:
                    epoch = int(row[0])
                    te_acc = float(row[4])
                    best = max(best, te_acc)
                except:
                    continue
    return best

def sweep_K(values=(1,2,4,8), base_name="K_sweep"):
    out_csv = os.path.join(PROJECT_ROOT, "reports", f"ablation_K.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = [["experiment","param_name","param_value","test_acc"]]
    for k in values:
        outdir = os.path.join(PROJECT_ROOT, "runs", base_name, f"K_{k}")
        cfg = {"coherence": {"K": int(k)}}
        acc = run_one(cfg, outdir)
        rows.append([base_name,"K",k, f"{acc:.4f}"])
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved:", out_csv)

def sweep_corr_px(values=(2.0,3.0,4.0,6.0), K=4, base_name="corr_sweep"):
    out_csv = os.path.join(PROJECT_ROOT, "reports", f"ablation_corr_px.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = [["experiment","param_name","param_value","test_acc"]]
    for c in values:
        outdir = os.path.join(PROJECT_ROOT, "runs", base_name, f"corr_{c}")
        cfg = {"coherence": {"K": int(K), "corr_px": float(c)}}
        acc = run_one(cfg, outdir)
        rows.append([base_name,"corr_px",c, f"{acc:.4f}"])
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved:", out_csv)

def sweep_shot(values=(500,100,50), base_name="shot_sweep"):
    out_csv = os.path.join(PROJECT_ROOT, "reports", f"ablation_shot.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = [["experiment","param_name","param_value","test_acc"]]
    for s in values:
        outdir = os.path.join(PROJECT_ROOT, "runs", base_name, f"shot_{s}")
        cfg = {"noise": {"shot_scale": float(s)}, "coherence": {"K": 1}}  # 建议先全相干对比
        acc = run_one(cfg, outdir)
        rows.append([base_name,"shot_scale",s, f"{acc:.4f}"])
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved:", out_csv)

def sweep_bits(values=("null",8,6,4), base_name="bits_sweep"):
    out_csv = os.path.join(PROJECT_ROOT, "reports", f"ablation_bits.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows = [["experiment","param_name","param_value","test_acc"]]
    for b in values:
        outdir = os.path.join(PROJECT_ROOT, "runs", base_name, f"bits_{b}")
        # YAML 里 null 才是 None；这里我们直接传 None 表示不量化
        qbits = None if (isinstance(b, str) and b.lower()=="null") else int(b)
        cfg = {"noise": {"quant_bits": qbits}, "coherence": {"K": 1}}
        acc = run_one(cfg, outdir)
        rows.append([base_name,"quant_bits",b, f"{acc:.4f}"])
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print("Saved:", out_csv)

if __name__ == "__main__":
    sweep_K(values=(1, 4, 8, 16, 32))
    sweep_corr_px(values=(0.5, 1.0, 1.5, 2.0), K=8)
    sweep_shot(values=(2000, 1000, 500, 100))
    #sweep_bits(values=("null",8,6,4))
    print("Done.")
