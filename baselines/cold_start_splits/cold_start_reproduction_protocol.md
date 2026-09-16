# Cold-Start Reproduction Protocol

This document specifies the exact cold-start evaluation protocol used in the current `LMDDA` project, so that external baseline implementations can reproduce the same experimental setting as closely as possible.

The intended use case is:

- the external team already has the same original datasets
- they need the exact split definition, seed policy, and evaluation setup
- they will run their own baseline models under the same cold-start setting

## 1. Purpose

We evaluate two cold-start settings:

1. `cold_drug`
2. `cold_disease`

These are stricter than the standard pair-level cross-validation setting because the held-out test entities are not seen in training.

The goal is to test generalization to:

- previously unseen drugs (`cold_drug`)
- previously unseen diseases (`cold_disease`)

## 2. Datasets

This protocol applies to the following datasets:

- `Bdataset`
- `Cdataset`
- `Fdataset`

Current dataset sizes in the final trainable version of our project are:

| Dataset | Drugs | Diseases | Positive Drug-Disease Pairs |
| --- | ---: | ---: | ---: |
| Bdataset | 269 | 598 | 18416 |
| Cdataset | 663 | 409 | 2532 |
| Fdataset | 593 | 313 | 1933 |

Important note:

- The cold-start split is defined on the **drug-disease pair table**.
- The pair table is constructed from the final binary drug-disease adjacency matrix used by the project.
- All possible drug-disease pairs are included in the evaluation universe, with labels:
  - `1` for known associations
  - `0` for unknown associations

## 3. Pair Table Construction

Let the drug-disease association matrix be `M` with shape:

- `N_drug x N_disease`

Create a full pair table with one row per `(drug_idx, disease_idx)` pair:

- `drug_idx`
- `disease_idx`
- `label`

where:

- `label = 1` if `M[drug_idx, disease_idx] == 1`
- `label = 0` otherwise

This means the cold-start protocol is performed over the **full matrix**, not only over positive samples.

## 4. Cross-Validation Overview

We use:

- `5-fold` outer cross-validation

For each outer fold:

- one subset of entities is assigned to `test`
- the remaining entities are further split into `train` and `val`

We do **not** use pair-level stratified splitting here.
Instead, the split unit is the entity type required by the cold-start condition.

## 5. Cold-Drug Split Definition

In `cold_drug`, the split unit is `drug_idx`.

### 5.1 Outer split

1. Take the sorted unique set of all drug indices:
   - `drug_ids = sorted(unique(drug_idx))`
2. Run:
   - `KFold(n_splits=5, shuffle=True, random_state=42)`
3. For one outer fold:
   - `test_drugs = drug_ids[test_indices]`
   - `train_val_drugs = drug_ids[train_val_indices]`

### 5.2 Inner split

Within the outer training drugs:

1. Run:
   - `KFold(n_splits=5, shuffle=True, random_state=42 + 777)`
2. Use the **first** inner split only
3. Define:
   - `train_drugs`
   - `val_drugs`

### 5.3 Pair assignment

Assign pair rows by drug membership:

- `train`: all rows whose `drug_idx in train_drugs`
- `val`: all rows whose `drug_idx in val_drugs`
- `test`: all rows whose `drug_idx in test_drugs`

This guarantees:

- no drug overlap between `train`, `val`, and `test`

Disease nodes are allowed to appear across all three sets.

## 6. Cold-Disease Split Definition

In `cold_disease`, the split unit is `disease_idx`.

### 6.1 Outer split

1. Take the sorted unique set of all disease indices:
   - `disease_ids = sorted(unique(disease_idx))`
2. Run:
   - `KFold(n_splits=5, shuffle=True, random_state=42)`
3. For one outer fold:
   - `test_diseases = disease_ids[test_indices]`
   - `train_val_diseases = disease_ids[train_val_indices]`

### 6.2 Inner split

Within the outer training diseases:

1. Run:
   - `KFold(n_splits=5, shuffle=True, random_state=42 + 777)`
2. Use the **first** inner split only
3. Define:
   - `train_diseases`
   - `val_diseases`

### 6.3 Pair assignment

Assign pair rows by disease membership:

- `train`: all rows whose `disease_idx in train_diseases`
- `val`: all rows whose `disease_idx in val_diseases`
- `test`: all rows whose `disease_idx in test_diseases`

This guarantees:

- no disease overlap between `train`, `val`, and `test`

Drug nodes are allowed to appear across all three sets.

## 7. Seed Policy

Use the following seeds exactly:

- outer split seed: `42`
- inner split seed: `819`  
  because our implementation uses `42 + 777 = 819`

For clarity:

- `outer KFold random_state = 42`
- `inner KFold random_state = 819`

If your implementation cannot mirror the exact "take the first inner split" behavior, document that clearly, because it changes the split definition.

## 8. Model Selection Within Each Fold

Within each outer fold:

1. Train only on the `train` subset
2. Use the `val` subset for model selection / early stopping / checkpoint selection
3. Report fold test metrics using the best checkpoint selected by `val`

This means:

- `test` is never used for model selection

## 9. Evaluation Universe

Evaluation is done on the full set of pair rows assigned to the split subset.

For example, in one test fold:

- evaluate on **all** test rows
- not only on positives
- not on sampled negatives

This is important:

- our protocol is a **full-matrix binary ranking / reconstruction setting**
- if a baseline instead evaluates on sampled negatives only, the results are not directly comparable

## 10. Metrics

Per fold, compute at least:

- `AUROC`
- `AUPR`

We recommend reporting:

- fold-wise `mean ± std`
- pooled out-of-fold test `AUROC`
- pooled out-of-fold test `AUPR`

### 10.1 Fold-wise reporting

For each of the 5 outer folds:

- evaluate on the fold-specific `test` rows
- compute `AUROC` and `AUPR`

Then report:

- mean across folds
- standard deviation across folds

### 10.2 Pooled reporting

After all 5 folds are complete:

1. concatenate the `test` predictions from the 5 folds
2. compute one global `AUROC`
3. compute one global `AUPR`

This pooled result is useful because it corresponds to a single out-of-fold prediction table over the entire dataset.

## 11. Prediction File Format

To ensure consistent downstream comparison, we recommend writing one test prediction file per fold with columns:

- `drug_idx`
- `disease_idx`
- `label`
- `score`
- `prob`

Where:

- `score` is the model output before thresholding
- `prob` is the predicted probability if available

If the baseline only produces one scalar score and not calibrated probabilities, that is acceptable, as long as the same score is used consistently for AUROC/AUPR.

## 12. Required Deliverables for Reproducibility

For each dataset and each split mode (`cold_drug`, `cold_disease`), the external baseline run should provide:

1. per-fold test metrics
   - `fold_0` to `fold_4`
   - `AUROC`, `AUPR`
2. aggregate metrics
   - mean ± std over 5 folds
3. pooled metrics
   - pooled `AUROC`
   - pooled `AUPR`
4. fold prediction files
   - one file per fold, containing test rows only

## 13. Recommended File Exchange

If the goal is to guarantee strict comparability, the best practice is:

### Preferred option

Provide the exact split assignments as files, rather than only a textual description.

For each dataset and split mode, provide a machine-readable split specification, for example:

- `fold_0_train.csv`
- `fold_0_val.csv`
- `fold_0_test.csv`

or one JSON file per dataset containing:

- fold id
- row indices of train/val/test

This is better than only giving the Markdown protocol because:

- it removes implementation ambiguity
- it prevents subtle differences in fold construction
- it makes baseline comparisons much more defensible in a paper

### Minimum acceptable option

If you do not want to exchange split files, then provide this protocol plus:

- the exact seeds
- the exact dataset version
- the exact pair-table generation rule

This is workable, but still less reliable than providing split files directly.

## 14. Should We Upload the Splits to GitHub?

Yes, in most cases this is the better choice.

If the other team is going to run baselines for direct comparison, the strongest setup is:

1. upload the exact cold-start split files to GitHub
2. upload this Markdown protocol
3. ask them to use your split files directly

That gives you:

- the cleanest comparability
- the least room for hidden mismatch
- the easiest future reuse for other baselines

My recommendation is:

- do **both**
  - provide the Markdown protocol
  - provide the exact split files

The protocol explains the logic.
The split files guarantee exact reproducibility.

## 15. Suggested Short Note to Collaborators

You can send the following note together with the files:

> We use entity-level 5-fold cold-start evaluation rather than pair-level stratified CV.  
> In `cold_drug`, train/val/test drugs are disjoint.  
> In `cold_disease`, train/val/test diseases are disjoint.  
> Evaluation is done on all held-out drug-disease pairs in the corresponding test subset, without sampled negatives.  
> To avoid implementation mismatch, please use the provided split files directly rather than reconstructing the folds independently.
