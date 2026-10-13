"""
Visualising the clustering results of the embeddings.


"""


from pathlib import Path
from typing import Any

import click
import numpy as np
import polars as pl
from mylib.cluster import labels_path, to_matrix
from mylib.viz.plots import plot_scatter
from mylib.viz.projection import project_2d


RUNS: list[tuple[str, dict[str, Any]]] = [
    ("kmeans", {"n_clusters": 5, "seed": 42}),
    ("hdbscan", {"min_cluster_size": 5}),
    ("hnne", {}),
]


def _id_column(df: pl.DataFrame, source: str) -> str:
    """Return the supported identifier column used by a data source."""
    for column in ("post_id", "instance_id"):
        if column in df.columns:
            return column
    raise click.ClickException(
        f"{source} must contain either 'post_id' or 'instance_id'; "
        f"found {df.columns}"
    )



@click.command()
@click.option("--embeddings_path", type=click.Path(exists=True, path_type=Path),
              default=Path("../data/synthetic_bio/embeddings/embeddings.parquet"),
              help="Path to the Parquet file containing embeddings.")
@click.option("--data_path",
              type=click.Path(exists=True, file_okay=True, path_type=Path),
              default=Path("../data/synthetic_bio/raw/synthetic_twitter_author_bios_200.csv"),
              help="Path to the input data."
)
@click.option("--clusters_dir",
              type=click.Path(path_type=Path),
              default=Path("../data/synthetic_bio/clusters"),
              help="Directory to save the output files."
)
@click.option("--viz_dir",
              type=click.Path(path_type=Path),
              default=Path("../data/synthetic_bio/viz"),
              help="Directory to save the output visualisations."
)
@click.option("--fmt", type=click.Choice(["parquet", "csv", "json"]),
              default="parquet", show_default=True, help="Format of the clusters + content table.")
@click.option("--only", type=str, help="Restrict to one corpus key.")
@click.option("--max_rows", type=int, help="Optional row cap for quick tests.")


def main(embeddings_path: Path, data_path: Path, clusters_dir: Path, viz_dir: Path,
         fmt: str, only: str | None, max_rows: int | None) -> None:
    """
    Save clusters with their content, and draw a map of each clustering.

    Args:
        embeddings_path: Folder with the saved embeddings.
        data_path: Folder with the deduplicated corpus (for text).
        cluster_dir: Folder with the saved cluster labels.
        viz_dir: Output folder.
        fmt: Format of the clusters + content table.
        only: Restrict to one corpus key.
        max_rows: Optional row cap for quick tests.

    Returns:
        None. Tables and maps are written under ``viz_dir``.
    """

    name = "synthetic_bio"
    if only and only != name:
        print(f"skipping {name}")
        return

    emb_df = pl.read_parquet(embeddings_path)
    data_df = pl.read_csv(data_path)
    if "embedding" not in emb_df.columns:
        raise click.ClickException(
            f"Embeddings must contain an 'embedding' column; found {emb_df.columns}"
        )
    if "authorbioDescription" not in data_df.columns:
        raise click.ClickException(
            "Input data must contain an 'authorbioDescription' column"
        )
    embedding_id = _id_column(emb_df, "Embeddings")
    data_id = _id_column(data_df, "Input data")

    if max_rows:
        emb_df = emb_df.head(max_rows)
    ids = emb_df[embedding_id].rename("id")
    coords = project_2d(
        to_matrix(emb_df, "embedding"),
        cache=viz_dir / "umap" / f"{name}{'_sub' if max_rows else ''}.npy",
    )
    text_df = data_df.select(
        pl.col(data_id).alias("id"),
        pl.col("authorbioDescription").fill_null(""),
    )
    out = viz_dir / "bio" / name
    out.mkdir(parents=True, exist_ok=True)

    for method, params in RUNS:
        labels_file = labels_path(clusters_dir, name, method, params)
        if not labels_file.exists():
            print(f"missing: {labels_file}")
            continue

        labels_df = pl.read_parquet(labels_file)
        table = (
            pl.DataFrame({"id": ids})
            .join(text_df, on="id", how="left")
            .join(labels_df, on="id", how="left")
            .with_columns(pl.col("cluster").fill_null(-1))
        )
        if table["cluster"].null_count():
            raise click.ClickException(f"No cluster label found for some rows in {labels_file}")
        if fmt == "parquet":
            table.write_parquet(out / f"{method}_clusters.parquet")
        elif fmt == "csv":
            table.write_csv(out / f"{method}_clusters.csv")
        else:
            table.write_ndjson(out / f"{method}_clusters.json")

        labels: np.ndarray = table["cluster"].to_numpy()
        hover: list[str] = table["authorbioDescription"].cast(pl.String).to_list()
        plot_scatter(
            coords,
            labels,
            hover=[text[:200] for text in hover],
            title=f"{name} / bio / {method}",
        ).write_html(out / f"{method}_map.html")
        print(f"{name} / bio / {method}: done ({(labels == -1).mean():.1%} noise)")


if __name__ == "__main__":
    main()