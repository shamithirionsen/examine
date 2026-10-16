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
import plotly.io as pio



RUNS: list[tuple[str, dict[str, Any]]] = [
    *[("kmeans", {"n_clusters": n, "seed": 42}) for n in (5, 10, 15, 20)],
    ("hnne", {}),
    ("hdbscan", {"min_cluster_size": 5}),
]

def _id_column(df: pl.DataFrame, source: str) -> str:
    """Return the supported identifier column used by a data source."""
    for column in ("post_id", "instance_id", "bio_id"):
        if column in df.columns:
            return column
    raise click.ClickException(
        f"{source} must contain either 'post_id' or 'instance_id'; "
        f"found {df.columns}"
    )


def _text_column(df: pl.DataFrame, source: Path) -> str:
    """Return the canonical or versioned bio description column."""
    if "bio_description" in df.columns:
        return "bio_description"
    candidates = [column for column in df.columns if column.startswith("bio_description_")]
    if len(candidates) == 1:
        return candidates[0]
    raise click.ClickException(
        f"{source} must contain one bio_description column; found {candidates}"
    )

def run(embeddings_path: Path, data_path: Path, clusters_dir: Path, viz_dir: Path,
        fmt: str = "parquet", only: str | None = None, max_rows: int | None = None,
        show: bool = False, version: str = "raw") -> None:
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
    version_label = version.replace("__", " + ").replace("_", " ")

    embedding_files = (
        [embeddings_path]
        if embeddings_path.is_file()
        else sorted(embeddings_path.rglob("*.parquet"))
    )
    if not embedding_files:
        raise click.ClickException(f"No Parquet embedding files found in {embeddings_path}")

    for embedding_file in embedding_files:
        emb_df = pl.read_parquet(embedding_file)
        if "embedding" not in emb_df.columns:
            raise click.ClickException(
                f"Embeddings must contain an 'embedding' column; found {emb_df.columns}"
            )
        embedding_id = _id_column(emb_df, "Embeddings")
        name = (
            embedding_file.relative_to(embeddings_path).with_suffix("")
            if embeddings_path.is_dir()
            else embedding_file.with_suffix("")
        )
        name_text = name.as_posix()
        if only and only != name_text:
            print(f"skipping {name_text}")
            continue

        data_file = (
            data_path
            if data_path.is_file()
            else data_path / f"{embedding_file.stem}.csv"
        )
        if not data_file.exists():
            raise click.ClickException(
                f"No matching raw data file found for {embedding_file}: {data_file}"
            )
        data_df = pl.read_csv(data_file)
        text_column = _text_column(data_df, data_file)
        data_id = _id_column(data_df, "Input data")
        text_df = data_df.select(
            pl.col(data_id).alias("id"),
            pl.col(text_column).fill_null("").alias("bio_description"),
        )

        if max_rows:
            emb_df = emb_df.head(max_rows)
        ids = emb_df[embedding_id].rename("id")
        coords = project_2d(
            to_matrix(emb_df, "embedding"),
            cache=viz_dir / "umap" / f"{name_text}{'_sub' if max_rows else ''}.npy",
        )
        tables_out = viz_dir / "tables" / "bio" / name
        maps_out = viz_dir / "maps" / "bio" / name
        tables_out.mkdir(parents=True, exist_ok=True)
        maps_out.mkdir(parents=True, exist_ok=True)

        for method, params in RUNS:
            labels_file = labels_path(clusters_dir, name_text, method, params)
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
            run_slug = "_".join(f"{key}{value}" for key, value in sorted(params.items()))
            run_name = f"{method}_{run_slug}" if run_slug else method
            if fmt == "parquet":
                table.write_parquet(tables_out / f"{run_name}_clusters.parquet")
            elif fmt == "csv":
                table.write_csv(tables_out / f"{run_name}_clusters.csv")
            else:
                table.write_ndjson(tables_out / f"{run_name}_clusters.json")

            labels: np.ndarray = table["cluster"].to_numpy()
            hover: list[str] = table["bio_description"].cast(pl.String).to_list()
            fig = plot_scatter(
                coords,
                labels,
                hover=[text[:200] for text in hover],
                title=f"{version_label} |{name_text} / bio / {run_name}",
            )
            if show:
                fig.show()
            fig.write_html(maps_out / f"{run_name}_map.html")
            print(f"{name_text} / bio / {run_name}: done ({(labels == -1).mean():.1%} noise)")


@click.command()
@click.option("--embeddings_path", type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
              default=Path("../data/synthetic_bio/embeddings"),
              help="Path to an embedding Parquet file or directory.")
@click.option("--data_path",
              type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
              default=Path("../data/synthetic_bio/raw"),
              help="Path to an input CSV file or directory of versioned CSV files."
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
@click.option("--show/--no-show", default=False, help="Display figures inline (notebooks).")
@click.option("--version", type=str, default="raw", help="Version name.")
def main(embeddings_path : Path, data_path: Path, clusters_dir: Path, viz_dir: Path, 
         fmt: str, only: str | None, max_rows: int | None, show: bool, version: str) -> None:
    
# def main(embeddings_path: Path, data_path: Path, clusters_dir: Path, viz_dir: Path,
        #  fmt: str, only: str | None, max_rows: int | None) -> None:
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
    run(embeddings_path, data_path, clusters_dir, viz_dir, fmt, only, max_rows, show, version)

    # embedding_files = (
    #     [embeddings_path]
    #     if embeddings_path.is_file()
    #     else sorted(embeddings_path.rglob("*.parquet"))
    # )
    # if not embedding_files:
    #     raise click.ClickException(f"No Parquet embedding files found in {embeddings_path}")

    # for embedding_file in embedding_files:
    #     emb_df = pl.read_parquet(embedding_file)
    #     if "embedding" not in emb_df.columns:
    #         raise click.ClickException(
    #             f"Embeddings must contain an 'embedding' column; found {emb_df.columns}"
    #         )
    #     embedding_id = _id_column(emb_df, "Embeddings")
    #     name = (
    #         embedding_file.relative_to(embeddings_path).with_suffix("")
    #         if embeddings_path.is_dir()
    #         else embedding_file.with_suffix("")
    #     )
    #     name_text = name.as_posix()
    #     if only and only != name_text:
    #         print(f"skipping {name_text}")
    #         continue

    #     data_file = (
    #         data_path
    #         if data_path.is_file()
    #         else data_path / f"{embedding_file.stem}.csv"
    #     )
    #     if not data_file.exists():
    #         raise click.ClickException(
    #             f"No matching raw data file found for {embedding_file}: {data_file}"
    #         )
    #     data_df = pl.read_csv(data_file)
    #     if "bio_description" not in data_df.columns:
    #         raise click.ClickException(
    #             f"Input data must contain a 'bio_description' column; found {data_df.columns}"
    #         )
    #     data_id = _id_column(data_df, "Input data")
    #     text_df = data_df.select(
    #         pl.col(data_id).alias("id"),
    #         pl.col("bio_description").fill_null(""),
    #     )

    #     if max_rows:
    #         emb_df = emb_df.head(max_rows)
    #     ids = emb_df[embedding_id].rename("id")
    #     coords = project_2d(
    #         to_matrix(emb_df, "embedding"),
    #         cache=viz_dir / "umap" / f"{name_text}{'_sub' if max_rows else ''}.npy",
    #     )
    #     tables_out = viz_dir / "tables" / "bio" / name
    #     maps_out = viz_dir / "maps" / "bio" / name
    #     tables_out.mkdir(parents=True, exist_ok=True)
    #     maps_out.mkdir(parents=True, exist_ok=True)

    #     for method, params in RUNS:
    #         labels_file = labels_path(clusters_dir, name_text, method, params)
    #         if not labels_file.exists():
    #             print(f"missing: {labels_file}")
    #             continue

    #         labels_df = pl.read_parquet(labels_file)
    #         table = (
    #             pl.DataFrame({"id": ids})
    #             .join(text_df, on="id", how="left")
    #             .join(labels_df, on="id", how="left")
    #             .with_columns(pl.col("cluster").fill_null(-1))
    #         )
    #         if table["cluster"].null_count():
    #             raise click.ClickException(f"No cluster label found for some rows in {labels_file}")
    #         run_slug = "_".join(f"{key}{value}" for key, value in sorted(params.items()))
    #         run_name = f"{method}_{run_slug}" if run_slug else method
    #         if fmt == "parquet":
    #             table.write_parquet(tables_out / f"{run_name}_clusters.parquet")
    #         elif fmt == "csv":
    #             table.write_csv(tables_out / f"{run_name}_clusters.csv")
    #         else:
    #             table.write_ndjson(tables_out / f"{run_name}_clusters.json")

    #         labels: np.ndarray = table["cluster"].to_numpy()
    #         hover: list[str] = table["bio_description"].cast(pl.String).to_list()
    #         # plot_scatter(
    #         #     coords,
    #         #     labels,
    #         #     hover=[text[:200] for text in hover],
    #         #     title=f"{name_text} / bio / {run_name}",
    #         # ).write_html(maps_out / f"{run_name}_map.html")

    #         # visualise the plot for a ipynb notebook
    #         fig = plot_scatter(
    #             coords,
    #             labels,
    #             hover=[text[:200] for text in hover],
    #             title=f"{name_text} / bio / {run_name}",
    #         )
    #         fig.show(renderer="notebook_connected")
    #         fig.write_html(maps_out / f"{run_name}_map.html")
    #         print(f"{name_text} / bio / {run_name}: done ({(labels == -1).mean():.1%} noise)")


if __name__ == "__main__":
    main()