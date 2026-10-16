# 



"""
1. Read data file 
2. Embed the text values. 
3. Save the embeddings to a Parquet file
"""


import polars as pl
from mylib.embed.embed import embed_texts, save_embeddings
import click
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Load environment variables from .env file
# DEFAULT_DATA_PATH = "../data/synthetic_bio/raw/synthetic_twitter_bios_300.csv" # raw/synthetic_twitter_bios_200.csv"
DEFAULT_DATA_FOLDER = "../data/synthetic_bio/processed"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def input_files(data_path: Path) -> list[Path]:
    """Return the CSV files represented by a file or directory path."""
    if data_path.is_file():
        if data_path.suffix.lower() != ".csv":
            raise click.BadParameter("the input file must be a CSV file")
        return [data_path]

    files = sorted(data_path.rglob("*.csv"))
    if not files:
        raise click.BadParameter(f"no CSV files found in {data_path}")
    return files


def text_column(df: pl.DataFrame, source: Path) -> str:
    """Return the canonical or versioned bio description column."""
    if "bio_description" in df.columns:
        return "bio_description"
    candidates = [column for column in df.columns if column.startswith("bio_description_")]
    if len(candidates) == 1:
        return candidates[0]
    raise click.ClickException(
        f"{source} must contain one bio_description column; found {candidates}"
    )


@click.command()
# @click.option("--data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default=DEFAULT_DATA_PATH, help="Path to the input CSV file.")
@click.option("--data-path", "--data-folder", "data_path",
              type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
              default=DEFAULT_DATA_FOLDER,
              help="Path to one CSV file or a directory of CSV files.")
@click.option("--embeddings_dir", type=click.Path(path_type=Path), default=Path("../data/synthetic_bio/embeddings"), help="Directory to save the output files.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Embedding model name.")
@click.option("--batch_size", default=128, help="Batch size for embedding requests.")
@click.option("--workers", default=8, help="Number of worker threads for concurrent embedding requests.")
def main(data_path: Path, embeddings_dir: Path, model: str,
         batch_size: int, workers: int) -> None:
    """Embed one CSV file or every CSV file in a directory."""
    files = input_files(data_path)
    output_dir = embeddings_dir / model.replace("/", "_")

    for data_file in files:
        source_df = pl.read_csv(data_file)
        text_name = text_column(source_df, data_file)
        df = source_df.select(["bio_id", text_name]).rename({text_name: "bio_description"})
        print(f"Embedding {len(df)} rows from {data_file} using model {model}...")
        texts = df["bio_description"].to_list()
        embeddings = embed_texts(texts, model=model, batch_size=batch_size, workers=workers)

        embedding_path = output_dir / f"{data_file.stem}.parquet"
        embedding_path.parent.mkdir(parents=True, exist_ok=True)
        save_embeddings(df["bio_id"].to_list(), embeddings, embedding_path)
        print(f"Saved {embedding_path} ({embeddings.shape[0]} embeddings)")
if __name__ == "__main__":
    main()


  