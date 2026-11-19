import math

import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/accuracy.csv")

# estimator ve n başına: ortalama hata, standart sapma
summary = (df.groupby(["estimator", "n"])["error"]
             .agg(["mean", "std"])
             .reset_index())
print(summary.to_string())

# teorik eğri
p_hat = math.pi / 4
ns = sorted(df["n"].unique())
theory = [4 * math.sqrt(p_hat * (1 - p_hat) / n) * 0.8 for n in ns]

fig, ax = plt.subplots()
for name, g in summary.groupby("estimator"):
    ax.plot(g["n"], g["mean"], marker="o", label=name)
ax.plot(ns, theory, "k--", label="teori (0.8·σ)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("n")
ax.set_ylabel("ortalama mutlak hata")
ax.legend()
ax.grid(True, which="both", alpha=0.3)
fig.savefig("results/accuracy.png", dpi=150)
print("kaydedildi: results/accuracy.png")