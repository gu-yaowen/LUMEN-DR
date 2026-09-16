# Embedding-Focused Ablation Study Plan

Goal: isolate how much the pretrained LM-derived initial node features help DDA prediction.

Reference model:
- tuned dataset-specific backbone/training settings
- pretrained drug, disease, and protein initial embeddings
- contrastive loss enabled

Ablation variants per dataset:
- `full_model`: pretrained drug + disease + protein, CL on
- `drug_id`: drug feature replaced by learnable ID embedding, disease/protein pretrained, CL on
- `disease_id`: disease feature replaced by learnable ID embedding, drug/protein pretrained, CL on
- `protein_id`: protein feature replaced by learnable ID embedding, drug/disease pretrained, CL on
- `no_cl`: all pretrained features kept, CL off

Implementation conventions:
- graph structure unchanged
- HGT backbone unchanged
- bilinear scorer unchanged
- only initial feature source and CL switch vary

CLI switches:
- `--drug-feature-source {pretrained,id}`
- `--disease-feature-source {pretrained,id}`
- `--protein-feature-source {pretrained,id}`
- `--use-contrastive {0,1}`

Practical note:
- “one-hot embedding” is implemented as a learnable ID embedding table per node type
- this keeps the codepath compatible with the current HGT pipeline
