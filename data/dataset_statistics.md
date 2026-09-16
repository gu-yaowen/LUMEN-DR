# Dataset Statistical Summary

This file summarizes the **current final trainable** versions of `Bdataset`, `Cdataset`, and `Fdataset` under `data/`.

## Definition

For a relation between two entity types with size `N1 x N2` and observed edge count `E`:

- `density = E / (N1 x N2)`
- `sparsity = 1 - density`

For `protein_protein`, density and sparsity are computed over `N_protein x N_protein` using the stored undirected edge list.

## Entity Counts

| Dataset | Drugs | Diseases | Proteins |
| --- | ---: | ---: | ---: |
| Bdataset | 269 | 598 | 6003 |
| Cdataset | 663 | 409 | 17800 |
| Fdataset | 593 | 313 | 17734 |

## Interaction Counts

| Dataset | Drug-Disease | Drug-Protein | Protein-Disease | Protein-Protein |
| --- | ---: | ---: | ---: | ---: |
| Bdataset | 18416 | 2107 | 17545 | 592926 |
| Cdataset | 2532 | 138775 | 13000 | 6356780 |
| Fdataset | 1933 | 136009 | 11819 | 6334988 |

## Density and Sparsity

### Drug-Disease

| Dataset | Density | Sparsity |
| --- | ---: | ---: |
| Bdataset | 0.114483222 | 0.885516778 |
| Cdataset | 0.009337419 | 0.990662581 |
| Fdataset | 0.010414366 | 0.989585634 |

### Drug-Protein

| Dataset | Density | Sparsity |
| --- | ---: | ---: |
| Bdataset | 0.001304800 | 0.998695200 |
| Cdataset | 0.011759198 | 0.988240802 |
| Fdataset | 0.012933208 | 0.987066792 |

### Protein-Disease

| Dataset | Density | Sparsity |
| --- | ---: | ---: |
| Bdataset | 0.004887467 | 0.995112533 |
| Cdataset | 0.001785665 | 0.998214335 |
| Fdataset | 0.002129265 | 0.997870735 |

### Protein-Protein

| Dataset | Density | Sparsity |
| --- | ---: | ---: |
| Bdataset | 0.016453709 | 0.983546291 |
| Cdataset | 0.020063060 | 0.979936940 |
| Fdataset | 0.020143382 | 0.979856618 |

## Notes

- These statistics are based on the final normalized training inputs currently used by the codebase.
- `Bdataset` protein-related files come from `REDDA`, then were sequence-cleaned and index-remapped.
- `Cdataset` and `Fdataset` were rebuilt using:
  - the curated existing `drug_protein` relations
  - CTD OMIM-based `gene-disease` subsets
  - STRING-based `GeneID -> StringProtein -> UniProt` remapping
  - STRING PPI subgraphs on the final retained protein sets
- In final `C/F`:
  - protein entity IDs are UniProt accessions
  - original gene identifiers are preserved in the `GeneID` column of `protein.csv`
- All three datasets pass strict loader validation with:
  - `mapping_related_drop_count = 0`
  - `runtime_missing_embedding_count = 0`
- All three datasets passed final training smoke tests:
  - `outputs/Bdataset/smoke_final_b`
  - `outputs/Cdataset/smoke_final_c`
  - `outputs/Fdataset/smoke_final_f`
