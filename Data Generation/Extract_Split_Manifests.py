"""Extract preserved train/test membership from the recovered joblib files."""

from __future__ import annotations

import json
from pathlib import Path

import joblib


DATA_ROOT = Path("data/training")
OUTPUT_ROOT = Path("data/splits")


def main() -> None:
    for basis_dir in sorted(path for path in DATA_ROOT.iterdir() if path.is_dir()):
        for molecule_dir in sorted(path for path in basis_dir.iterdir() if path.is_dir()):
            scaled_dir = molecule_dir / "scaled"
            train_path = scaled_dir / "full_feature_train.bin"
            test_path = scaled_dir / "full_feature_test.bin"
            if not train_path.exists() or not test_path.exists():
                continue

            train_names = list(joblib.load(train_path)["names"])
            test_names = list(joblib.load(test_path)["names"])
            overlap = sorted(set(train_names) & set(test_names))
            if overlap:
                raise ValueError(
                    f"{basis_dir.name}/{molecule_dir.name} has overlapping split names: "
                    f"{overlap[:5]}"
                )

            manifest = {
                "basis_set": basis_dir.name,
                "molecule": molecule_dir.name,
                "source": "recovered full_feature_train.bin/full_feature_test.bin metadata",
                "name_semantics": (
                    "Zero-based index into the ordered pickle sequence; index 0 "
                    "corresponds to the molecule2_alldata.pkl.gz file."
                ),
                "train_names": train_names,
                "test_names": test_names,
            }

            output_path = OUTPUT_ROOT / basis_dir.name / f"{molecule_dir.name}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
