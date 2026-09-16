# Feature caches

LUMEN-DR consumes frozen entity-level embeddings rather than running the
pretrained encoders inside the graph-training loop. The expected files are:

```text
features/<DATASET>/MolFormer_drug_emb.pkl
features/<DATASET>/BERT_disease_emb.pkl
features/<DATASET>/ESMC_protein_emb.pkl
```

Each file is a pickled dictionary keyed by the entity identifier used in the
matching `data/<DATASET>/*.csv` file. The loader validates feature coverage and
converts the dictionaries to tensors in `src/lmdda/utils/features.py`.

The original feature-generation script and model checkpoints are not included
in this cleaned repository. To regenerate the caches, use the exact pretrained
model versions and preprocessing rules recorded in the paper and in
`data/dataset_process.md`:

- MolFormer for canonical drug SMILES;
- BioBERT for the formatted disease name and description text;
- ESMC for protein sequences, mean-pooled after removing special tokens.

Keep the generated caches outside version control. Before training, validate
that every released drug, disease, and protein ID has exactly one embedding and
that the embedding dimensions are consistent within each entity type.
