"""
Clusterise embeddings using different clustering methods and parameters.


"""

import click 
from mylib.cluster import cluster, save_labels, to_matrix
from typing import Any
from pathlib import Path
import polars as pl



# One entry per run: (method, params). Add or remove cluster counts here.
KMEANS_CLUSTERS = (5, 10, 15, 20)
RUNS: list[tuple[str, dict[str, Any]]] = [
    *[("kmeans", {"n_clusters": n, "seed": 42}) for n in KMEANS_CLUSTERS],
]
RUNS += [
    ("hdbscan", {"min_cluster_size": 5}),
    ("hnne", {}),
]


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


@click.command()
@click.option("--embeddings_path", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
              default=Path("../data/synthetic_bio/embeddings"), help="Path to an embedding Parquet file or directory.")
@click.option("--clusters_dir", type=click.Path(path_type=Path), default=Path("../data/synthetic_bio/clusters"), help="Directory to save the output files.")


def main(embeddings_path: Path, clusters_dir: Path) -> None:
    """Cluster embeddings using different methods and parameters, and save the results."""
    embedding_files = (
        [embeddings_path]
        if embeddings_path.is_file()
        else sorted(embeddings_path.rglob("*.parquet"))
    )
    if not embedding_files:
        raise click.ClickException(f"No Parquet embedding files found in {embeddings_path}")

    # Create output directories
    clusters_dir.mkdir(parents=True, exist_ok=True)

    for embedding_file in embedding_files:
        df = pl.read_parquet(embedding_file)
        ids: pl.Series = df[_id_column(df)]
        emb = to_matrix(df, "embedding")
        name = (
            embedding_file.relative_to(embeddings_path).with_suffix("")
            if embeddings_path.is_dir()
            else embedding_file.with_suffix("")
        )

        # Perform clustering for each embedding file and run.
        for method, params in RUNS:
            print(f"Clustering {name} with {method} and parameters {params}...")
            labels = cluster(emb, method=method, **params)
            save_labels(ids, labels, clusters_dir, name.as_posix(), method, params)


if __name__ == "__main__":
    main()