import matplotlib.pyplot as plt

from analyze_speedup import make_table

CHUNKS = [1_000_000, 100_000, 10_000]
PS = [1, 2, 4, 6, 8, 12]


def main():
    fig, ax = plt.subplots()
    for c in CHUNKS:
        t = make_table(f"results/chunk_{c}.csv", "numpy_serial")
        ax.plot(t["workers"], t["speedup"], marker="o", label=f"NumPy chunk={c:,}")

    t_pu = make_table("results/speedup_pure.csv", "pure_serial")
    ax.plot(t_pu["workers"], t_pu["speedup"], marker="s", linestyle="--",
            color="gray", label="Pure Python")
    ax.plot(PS, PS, "k:", label="ideal")
    ax.axvline(6, color="gray", linestyle=":", alpha=0.4)
    ax.text(6.1, 11, "6 fiziksel çekirdek", fontsize=8, color="gray")
    ax.set_xlabel("işçi sayısı (p)")
    ax.set_ylabel("speedup (aynı chunk'lı seriye göre)")
    ax.set_xticks(PS)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig("results/speedup_chunk.png", dpi=150)
    print("kaydedildi: results/speedup_chunk.png")


if __name__ == "__main__":
    main()