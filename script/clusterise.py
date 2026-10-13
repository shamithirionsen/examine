"""
Clusterise embeddings using different clustering methods and parameters.
This script performs clustering on embeddings using KMeans, HDBSCAN, and HNNE methods
Saves in csv in the output directory:
- cluster labels for each method and parameter combination
- cluster sizes for each method and parameter combination
Saves in the viz directory:
- cluster size bar charts for each method and parameter combination
- contingency heatmaps for each pair of methods and parameter combinations
The output directory structure is as follows:

output_dir/
    ├── clusters/
    │   ├── kmeans_n_clusters_20_seed_42.csv
    │   ├── hdbscan_min_cluster_size_50.csv
    │   └── hnne.csv
    └── viz/
        ├── cluster_sizes/
        │   ├── kmeans_n_clusters_20_seed_42.png
        │   ├── hdbscan_min_cluster_size_50.png
        │   └── hnne.png
        └── contingency_heatmaps/
            ├── kmeans_n_clusters_20_seed_42_vs_hdbscan_min_cluster_size_50.png
            ├── kmeans_n_clusters_20_seed_42_vs_hnne.png
            └── hdbscan_min_cluster_size_50_vs_hnne.png

"""

import click 
from mylib.cluster import cluster, save_labels, to_matrix
from typing import Any
from pathlib import Path
import polars as pl



# One entry per run: (method, params). Add or remove runs here.
RUNS: list[tuple[str, dict[str, Any]]] = [
    ("kmeans", {"n_clusters": 5, "seed": 42}),
    ("hdbscan", {"min_cluster_size": 5}),
    ("hnne", {}),
]


@click.command()
def _id_column(df: pl.DataFrame) -> str:
    """Return the supported identifier column used by the embeddings.
    Normalises the identifier column name to either 'post_id' or 'instance_id' 
    for consistency across different data sources.

    Args:
        df: A Polars DataFrame containing the embeddings.
    Returns:
        The name of the identifier column ('post_id' or 'instance_id').
    Raises:
        click.ClickException: If neither 'post_id' nor 'instance_id' is found in the 
        DataFrame columns.
    """
    for column in ("post_id", "instance_id"):
        if column in df.columns:
            return column
    raise click.ClickException(
        f"Embeddings must contain either 'post_id' or 'instance_id'; found {df.columns}"
    )


@click.option("--embeddings_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), 
              default=Path("../data/synthetic_bio/embeddings/embeddings.parquet"), help="Path to the input Parquet file containing embeddings.")
@click.option("--clusters_dir", type=click.Path(path_type=Path), default=Path("../data/synthetic_bio/clusters"), help="Directory to save the output files.")


def main(embeddings_path: Path, clusters_dir: Path) -> None:
    """Cluster embeddings using different methods and parameters, and save the results."""
    # Load embeddings from Parquet file
    df = pl.read_parquet(embeddings_path)
    ids = df.select(_id_column(df)).to_series().to_list()
    emb = to_matrix(df, "embedding")

    # Create output directories
    clusters_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = clusters_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Perform clustering for each run
    for method, params in RUNS:
        print (f"Clustering with {method} and parameters {params}...")
        labels = cluster(emb, method=method, **params)
        save_labels(ids, labels, clusters_dir, "synthetic_bio", method, params)


if __name__ == "__main__":
    main()