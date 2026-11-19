import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/weak.csv")
par = df[df["estimator"] == "parallel_numpy"]
g = par.groupby("workers")["elapsed"]
med, lo, hi = g.median(), g.min(), g.max()
t1 = med[1]
eff = t1 / med

table = pd.DataFrame({"n": par.groupby("workers")["n"].first(),
                      "median_s": med, "efficiency": eff,
                      "scaled_speedup": eff * med.index})
print(table.round(4).to_string())

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
ps = list(med.index)
a1.errorbar(ps, med, yerr=[med - lo, hi - med], marker="o", capsize=3)
a1.axhline(t1, color="k", linestyle="--", label="ideal (sabit süre)")
a1.axvline(6, color="gray", linestyle=":", alpha=0.5)
a1.set_xlabel("işçi sayısı (p)")
a1.set_ylabel("süre (s), min-medyan-max")
a1.set_xticks(ps)
a1.legend()
a1.grid(True, alpha=0.3)

a2.plot(ps, eff, marker="o")
a2.axhline(1, color="k", linestyle="--")
a2.axvline(6, color="gray", linestyle=":", alpha=0.5)
a2.set_xlabel("işçi sayısı (p)")
a2.set_ylabel("weak scaling verimliliği T(1)/T(p)")
a2.set_xticks(ps)
a2.set_ylim(0, 1.1)
a2.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig("results/weak.png", dpi=150)
print("kaydedildi: results/weak.png")