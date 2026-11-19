import matplotlib.pyplot as plt
import pandas as pd

from pimc.metrics import speedup, efficiency


def make_table(csv_path: str, serial_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    med = df.groupby(["estimator", "workers"])["elapsed"].median()
    t_serial = med[(serial_name, 1)]

    rows = []
    for (est, p), t in med.items():
        if est == serial_name:
            continue
        rows.append({
            "workers": p,
            "median_s": t,
            "speedup": speedup(t_serial, t),
            "efficiency": efficiency(t_serial, t, p),
        })
    table = pd.DataFrame(rows).sort_values("workers").reset_index(drop=True)
    print(f"\n{csv_path}  (seri medyan: {t_serial:.4f} s)")
    print(table.to_string(index=False))
    return table


def main():
    t_np = make_table("results/speedup_numpy.csv", "numpy_serial")
    t_pu = make_table("results/speedup_pure.csv", "pure_serial")

    fig, ax = plt.subplots()
    ax.plot(t_np["workers"], t_np["speedup"], marker="o", label="NumPy (n=1e8)")
    ax.plot(t_pu["workers"], t_pu["speedup"], marker="s", label="Pure Python (n=1e7)")
    ps = [1, 2, 4, 6, 8, 12]
    ax.plot(ps, ps, "k--", label="ideal")
    ax.axvline(6, color="gray", linestyle=":", alpha=0.6)
    ax.text(6.1, 0.5, "6 fiziksel çekirdek", fontsize=8, color="gray")
    ax.set_xlabel("işçi sayısı (p)")
    ax.set_ylabel("speedup")
    ax.set_xticks(ps)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig("results/speedup.png", dpi=150)
    print("\nkaydedildi: results/speedup.png")


if __name__ == "__main__":
    main()