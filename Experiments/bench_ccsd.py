"""Reference wall times for HF, MP2 and frozen-core CCSD with Psi4 (single thread).

SCF settings mirror Data Generation/frozen_core_CCSD_Clean.py (scf_type pk, mp2_type conv,
e/d_convergence 1e-8). Each timing includes its own SCF.
Usage: python -I bench_ccsd.py <geometry_root> <out.json>
"""
import json
import sys
import time
from pathlib import Path

import psi4

geo, out = Path(sys.argv[1]), Path(sys.argv[2])
jobs = [
    ("Water", geo / "training" / "water" / "water100.xyz", "STO-3G"),
    ("Water", geo / "training" / "water" / "water100.xyz", "cc-pVDZ"),
    ("Formaldehyde", geo / "training" / "formaldehyde" / "formaldehyde100.xyz", "STO-3G"),
    ("Formaldehyde", geo / "training" / "formaldehyde" / "formaldehyde100.xyz", "cc-pVDZ"),
    ("Pentane", geo / "blind" / "Pentane.xyz", "STO-3G"),
    ("Benzene", geo / "blind" / "Benzene.xyz", "STO-3G"),
    ("Benzene", geo / "blind" / "Benzene.xyz", "cc-pVDZ"),
]
psi4.core.be_quiet()
psi4.set_num_threads(1)
psi4.set_memory("8 GB")
rows = []
for name, path, basis in jobs:
    text = path.read_text()
    if "units" not in text:
        qmol = psi4.qcdb.Molecule.from_string(text, dtype="xyz")
        text = qmol.create_psi4_string_from_molecule()
    row = {"molecule": name, "basis": basis}
    for method in ("scf", "mp2", "ccsd"):
        psi4.core.clean()
        extra = "\nno_reorient\nno_com" if "symmetry" in text else "\nsymmetry c1\nno_reorient\nno_com"
        mol = psi4.geometry(text.rstrip() + extra)
        psi4.set_options({"basis": basis, "scf_type": "pk", "mp2_type": "conv", "freeze_core": True,
                          "e_convergence": 1e-8, "d_convergence": 1e-8})
        t0 = time.perf_counter()
        e, wfn = psi4.energy(method, return_wfn=True, molecule=mol)
        row[f"{method}_s"] = time.perf_counter() - t0
        row[f"{method}_E"] = e
        if method == "scf":
            nfz = wfn.basisset().n_frozen_core()
            nocc = wfn.nalpha() - nfz
            nvir = wfn.nmo() - wfn.nalpha()
            row.update({"nbf": wfn.nmo(), "n_active_occ": nocc, "n_virt": nvir,
                        "n_t2_amplitudes": nocc ** 2 * nvir ** 2})
    print(row, flush=True)
    rows.append(row)
out.write_text(json.dumps(rows, indent=2))
