"""LaTeX bodies of SI Tables S4 (formaldehyde-trained models under different tuning protocols) and
S5 (kernel-bandwidth scan), generated from the CSV outputs of matched_transfer.py, cobyla_arm.py and
gamma_scan.py. Row groups follow Tables S4 and S5.

Usage: python -I make_tables_s4_s5.py <workdir>
"""
import re
import sys
from pathlib import Path

import pandas as pd

work = Path(sys.argv[1])
TG = ["Water", "Ammonia", "Methane", "Ethylene", "Methanol", "Ethane"]
COLS = ["Formaldehyde (test)"] + TG
NAMES = {"classical": "Classical (RBF)", "Hubregtsen": "Hubregtsen", "MultiControl": "MultiControl",
         "Chebyshev": "ChebyshevPQC"}
ENC = {"0": "original", "1": "degree 1", "2": "degree 2", "atan": "arctan"}


def fmt(v):
    return f"${v:.2f}$" if v < 0 else f"{v:.2f}"


def config_str(row):
    if row["family"] == "classical":
        return f"$\\gamma={row['gamma']:g}$, $C={row['C']:g}$, $\\epsilon={row['epsilon']:g}$"
    m = re.match(r"(\w+?)_L(\d)_(\w+)", row["config"])
    return (f"$L={m.group(2)}$, {ENC[m.group(3)]}, $\\gamma={row['gamma']:g}$, "
            f"$C={row['C']:g}$, $\\epsilon={row['epsilon']:g}$")


fin = pd.read_csv(work / "final_results_basegrid.csv")
cob = pd.read_csv(work / "cobyla_results.csv") if (work / "cobyla_results.csv").exists() else None
rows = []
for prot, label in [("paper", "Reference grid (conformer-grouped cross-validation)"), ("matched", "Matched tuning")]:
    for fam in NAMES:
        sel = fin[(fin.family == fam) & (fin.protocol == prot) & (fin.shots.isin(["none", "shots1024"]))]
        if len(sel):
            r = sel.iloc[0]
            rows.append((label, fam, config_str(r), r[COLS]))
for fam in NAMES:  # exact expectation values: matched selection and the same configuration + COBYLA
    sel = fin[(fin.family == fam) & (fin.protocol == "matched") & (fin.shots == "exact")]
    if len(sel):
        r = sel.iloc[0]
        rows.append(("Matched tuning, exact expectation values", fam, config_str(r), r[COLS]))
if cob is not None:
    for fam in NAMES:
        sel = cob[(cob.family == fam) & (cob.shots == "exact")]
        if len(sel):
            r = sel.iloc[0]
            rows.append(("Matched tuning and COBYLA, exact expectation values", fam, config_str(r), r[COLS]))

print("% ---------- Table S4 body ----------")
last = None
for label, fam, cfg, vals in rows:
    if label != last:
        if last is not None:
            print(r"        \midrule")
        print(rf"        \multicolumn{{9}}{{l}}{{\emph{{{label}}}}} \\")
        last = label
    v = [float(vals[c]) for c in COLS]
    mean = sum(v[1:]) / 6
    print(f"        {NAMES[fam]} & " + " & ".join(fmt(x) for x in v) + f" & {fmt(mean)} \\\\")
print("% configurations:")
for label, fam, cfg, _ in rows:
    print(f"%   {label} | {NAMES[fam]} | {cfg}")

gs = pd.read_csv(work / "gamma_scan.csv")
print("% ---------- Table S5 body ----------")
for g in sorted(gs.gamma.unique()):
    cells = []
    for fam in NAMES:
        s = gs[(gs.family == fam) & (gs.gamma == g)]
        cells += ["--", "--"] if s.empty else [f"{s.iloc[0].cv_r2:.3f}", fmt(s.iloc[0].mean_targets)]
    print(f"        {g:g} & " + " & ".join(cells) + r" \\")
