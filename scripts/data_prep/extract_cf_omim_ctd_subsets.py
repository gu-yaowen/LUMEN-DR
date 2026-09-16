from __future__ import annotations

import csv
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CTD_GZ = DATA_DIR / "reference" / "ctd" / "CTD_genes_diseases.tsv.gz"
CTD_DIR = DATA_DIR / "reference" / "ctd"

OUTPUTS = {
    "Cdataset": CTD_DIR / "CTD_genes_diseases_omim_Cdataset.csv",
    "Fdataset": CTD_DIR / "CTD_genes_diseases_omim_Fdataset.csv",
}

MESH_DERIVED_FILES = [
    CTD_DIR / "CTD_genes_diseases_mesh.csv",
    CTD_DIR / "CTD_genes_diseases_mesh_unique_disease_ids.txt",
    CTD_DIR / "CTD_genes_diseases_mesh_Cdataset_direct.csv",
    CTD_DIR / "CTD_genes_diseases_mesh_Fdataset_direct.csv",
    CTD_DIR / "CTD_genes_diseases_mesh_Cdataset_all.csv",
    CTD_DIR / "CTD_genes_diseases_mesh_Fdataset_all.csv",
]


def load_dataset_omim_ids(dataset_name: str) -> set[str]:
    disease_csv = DATA_DIR / dataset_name / "disease.csv"
    omim_ids: set[str] = set()
    with disease_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease_id = (row.get("Disease") or "").strip()
            if disease_id.startswith("D") and len(disease_id) > 1:
                omim_ids.add(f"OMIM:{disease_id[1:]}")
    return omim_ids


def remove_old_mesh_files() -> list[str]:
    removed: list[str] = []
    for path in MESH_DERIVED_FILES:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    return removed


def main() -> None:
    dataset_omim = {
        dataset: load_dataset_omim_ids(dataset) for dataset in OUTPUTS
    }

    writers = {}
    handles = {}
    counts = {dataset: 0 for dataset in OUTPUTS}

    try:
        with gzip.open(CTD_GZ, "rt", encoding="utf-8", newline="") as source:
            header = None
            for raw_line in source:
                if raw_line.startswith("#"):
                    continue
                header = next(csv.reader([raw_line]))
                break
            if header is None:
                raise RuntimeError("Failed to read CTD header from genes_diseases file.")

            for dataset, output_path in OUTPUTS.items():
                handle = output_path.open("w", encoding="utf-8", newline="")
                handles[dataset] = handle
                writer = csv.writer(handle)
                writers[dataset] = writer
                writer.writerow(header)

            for row in csv.reader(source, delimiter="\t"):
                if not row:
                    continue
                disease_id = row[3].strip()
                if not disease_id.startswith("OMIM:"):
                    continue
                for dataset, valid_ids in dataset_omim.items():
                    if disease_id in valid_ids:
                        writers[dataset].writerow(row)
                        counts[dataset] += 1
    finally:
        for handle in handles.values():
            handle.close()

    removed = remove_old_mesh_files()

    print("Counts:")
    for dataset, count in counts.items():
        print(f"{dataset}\t{count}\t{OUTPUTS[dataset]}")
    print("Removed:")
    for path in removed:
        print(path)


if __name__ == "__main__":
    main()
