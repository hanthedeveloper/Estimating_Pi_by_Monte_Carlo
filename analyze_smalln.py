import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/smalln.csv")
med = df.groupby(["estimator", "workers", "n"])["elapsed"].median()

ns = sorted(df["n"].unique())
t_serial = {n: med[("numpy_serial", 1, n)] for n in ns}

fig, ax = plt.subplots()
rows = []
for p in [2, 4, 6, 12]:
    sp = [t_serial[n] / med[("parallel_numpy", p, n)] for n in ns]
    ax.plot(ns, sp, marker="o", label=f"p={p}")
    for n, s in zip(ns, sp):
        rows.append({"p": p, "n": n, "speedup": round(s, 2)})

ax.axhline(1, color="k", linestyle="--", label="break-even (speedup=1)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("n")
ax.set_ylabel("speedup (aynı chunk'lı seriye göre)")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.savefig("results/smalln.png", dpi=150)

print(pd.DataFrame(rows).pivot(index="n", columns="p", values="speedup"))
print("kaydedildi: results/smalln.png")