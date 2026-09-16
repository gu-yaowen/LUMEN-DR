# Dataset Processing Notes

This file records the current final data-preparation workflow used by `LMDDA` for `Bdataset`, `Cdataset`, and `Fdataset`.

## 1. Data sources

- Base drug-disease datasets were originally taken from `LLM_DDA`.
- `Bdataset` protein-related files were rebuilt using `REDDA/dataset/Bdataset`.
- `Cdataset` and `Fdataset` were rebuilt from:
  - existing curated `drug_protein.csv`
  - CTD `gene-disease` records filtered by OMIM disease IDs
  - STRING human aliases and links for protein remapping and PPI subgraph extraction

Reference files are stored in:

- `data/reference/ctd/`
- `data/reference/string/`

## 2. Multimodal entity information

Final entity modalities used by the project:

- Drugs: `SMILES`
- Diseases: textual `description`
- Proteins: amino-acid `Sequence`

Cached pretrained feature files are stored in:

- `features/<DATASET>/MolFormer_drug_emb.pkl`
- `features/<DATASET>/BERT_disease_emb.pkl`
- `features/<DATASET>/ESMC_protein_emb.pkl`

## 3. Protein sequence supplementation

Protein sequences were filled before final training-data normalization.

- `Bdataset`: missing proteins were queried from UniProt by accession.
- `Cdataset` and `Fdataset`: missing proteins were supplemented through:
  - STRING sequence matching
  - NCBI `gene2refseq`
  - additional `GeneID -> UniProt` mapping

After sequence collection:

- `Bdataset` still had `37` proteins without valid sequence support and these were removed during final cleanup.
- `Cdataset` and `Fdataset` were restricted to the sequence-supported protein subsets used for final training.

## 4. Final training-input normalization

The current training code assumes a strict normalized format:

- each entity file contains a contiguous `ID` column from `0..N-1`
- `drug_dis.csv` row/column order matches `drug.csv` / `disease.csv`
- all edge files store contiguous training indices
- all training entities have matching cached embeddings

The Bdataset normalization script also requires the original REDDA Bdataset
files. Set `LUMEN_DR_REDDA_ROOT` to that external dataset directory before
running `normalize_training_datasets.py`; otherwise it defaults to
`external/REDDA/dataset/Bdataset` relative to the repository root.

## 5. Dataset-specific final processing

### 5.1 Bdataset

For `Bdataset`:

- `drug.csv` and `disease.csv` order was verified against `REDDA`
- protein-related files were rebuilt from `REDDA/dataset/Bdataset`
- proteins without valid sequence/embedding support were removed
- `drug_protein.csv`, `protein_disease.csv`, and `protein_protein.csv` were remapped into the final contiguous protein index space

Final trainable `Bdataset` uses:

- `6003` proteins
- `2107` drug-protein edges
- `17545` protein-disease edges
- `592926` protein-protein edges

### 5.2 Cdataset and Fdataset

The final `C/F` rebuild was updated after confirming that their disease IDs are OMIM-style IDs rather than MeSH IDs.

The final procedure was:

1. Read `disease.csv` and convert each disease ID from `Dxxxxxx` to `OMIM:xxxxxx`
2. Extract CTD `gene-disease` records whose `DiseaseID` matches those OMIM IDs
3. Decode the existing curated `drug_protein.csv` back to `GeneID` using the previous `protein.csv`
4. Merge the `drug-protein` gene set with the CTD OMIM `gene-disease` gene set
5. Keep only proteins that satisfy all of the following:
   - have non-empty sequence
   - have a valid `StringProtein`
   - can be mapped to a UniProt accession through STRING alias data
6. Rebuild:
   - `protein.csv`
   - `drug_protein.csv`
   - `protein_disease.csv`
   - `protein_protein.csv`
7. Rewrite `ESMC_protein_emb.pkl` so that feature keys match the final protein entity IDs

Important final convention for `C/F`:

- `protein.csv` now uses **UniProt accession** as the final `Protein` entity ID
- the original CTD/NCBI gene identifier is preserved in the `GeneID` column

Intermediate OMIM CTD subsets are stored in:

- `data/reference/ctd/CTD_genes_diseases_omim_Cdataset.csv`
- `data/reference/ctd/CTD_genes_diseases_omim_Fdataset.csv`

Final rebuild report:

- `reports/cf_rebuild_from_existing_dp_and_omim_report.json`

Final trainable `Cdataset`:

- input drug-protein gene edges: `139955`
- input OMIM gene-disease edges: `15391`
- final proteins: `17800`
- final drug-protein edges: `138775`
- final protein-disease edges: `13000`
- final protein-protein edges: `6356780`

Final trainable `Fdataset`:

- input drug-protein gene edges: `137155`
- input OMIM gene-disease edges: `14121`
- final proteins: `17734`
- final drug-protein edges: `136009`
- final protein-disease edges: `11819`
- final protein-protein edges: `6334988`

For both `Cdataset` and `Fdataset`:

- final protein feature files were rewritten successfully
- `protein_feature_missing_old_embedding = 0`

## 6. Final validation

All three current datasets pass strict loader validation:

- `mapping_related_drop_count = 0`
- `runtime_missing_embedding_count = 0`
- all entity IDs are contiguous
- all edge indices fall within valid ranges

Validation reports:

- `reports/training_input_normalization_report.json`
- `reports/training_input_validation_report.json`

## 7. Smoke-test status

The current HGT training pipeline was re-run on the final trainable versions of all three datasets.

Successful final smoke runs:

- `outputs/Bdataset/smoke_final_b`
- `outputs/Cdataset/smoke_final_c`
- `outputs/Fdataset/smoke_final_f`

These runs confirm that the current training code can:

- load all entities, features, and edges
- construct the heterograph successfully
- train for at least one epoch with cross-validation
- save checkpoints, metrics, predictions, and logs without index or feature-alignment failures
