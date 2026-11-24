"""Kernel-bandwidth scan for formaldehyde-trained models.

For every gamma of the matched grid, the (C, epsilon) with the best conformer-grouped CV R^2 on
formaldehyde is refitted on all formaldehyde training conformers and evaluated on the other molecules.
Classical RBF-SVR and the three quantum circuits in the configuration of the main text
(L=1, original encoding, 1024-shot noise) are scanned.

Usage: python -I gamma_scan.py <workdir> [n_jobs]
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matched_transfer as mt  # noqa: E402

work = Path(sys.argv[1])
n_jobs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
jobs = []
for fname, family, shots in [("cv_classical_none.csv", "classical", "none"),
                             ("cv_Hubregtsen_L1_0_shots1024.csv", "Hubregtsen", "shots1024"),
                             ("cv_MultiControl_L1_0_shots1024.csv", "MultiControl", "shots1024"),
                             ("cv_Chebyshev_L1_0_shots1024.csv", "Chebyshev", "shots1024")]:
    d = pd.read_csv(work / fname)
    best = d.loc[d.groupby("gamma")["cv_r2"].idxmax()]
    for _, row in best.iterrows():
        jobs.append((family, f"gamma-scan", shots, row))
res = Parallel(n_jobs=n_jobs)(delayed(mt._final_one)(work, fam, prot, sm, row) for fam, prot, sm, row in jobs)
df = pd.DataFrame(res).sort_values(["family", "gamma"])
targets = ["Water", "Ammonia", "Methane", "Ethylene", "Methanol", "Ethane"]
df["mean_targets"] = df[targets].mean(axis=1)
df.to_csv(work / "gamma_scan.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(df[["family", "gamma", "C", "epsilon", "cv_r2", "n_sv", "Formaldehyde (test)"] + targets
             + ["mean_targets"]].round(3).to_string(index=False))
