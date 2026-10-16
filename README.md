## Examine

The experiment has one pipeline:

1. Embed each configured text version and model.
2. Cluster every embedding file with the configured runs.
3. Build cluster tables and 2D visualizations.

Run it from the repository root with:

```bash
uv run examine
```

Use a different configuration or recompute existing embeddings with:

```bash
uv run examine --config experiment.yaml --overwrite
```

The configured data root contains `processed/`, `embeddings/`, `clusters/`, and
`viz/` directories. The canonical implementation lives in `src/examine`.
