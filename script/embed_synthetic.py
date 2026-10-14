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
DEFAULT_DATA_PATH = "../data/synthetic_bio/raw/synthetic_twitter_bios_300.csv" # raw/synthetic_twitter_bios_200.csv"
MODEL =  "Qwen/Qwen3-Embedding-0.6B"  # "BAAI/bge-m3" # 2 for now, think of a dictionary for later


@click.command()
@click.option("--data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), default=DEFAULT_DATA_PATH, help="Path to the input CSV file.")
@click.option("--embeddings_dir", type=click.Path(path_type=Path), default=Path("../data/synthetic_bio/embeddings"), help="Directory to save the output files.")
@click.option("--batch_size", default=128, help="Batch size for embedding requests.")
@click.option("--workers", default=8, help="Number of worker threads for concurrent embedding requests.")
def main(data_path: str, embeddings_dir: Path, batch_size: int, workers: int) -> None:
    """Read data file, embed the text values, and save the embeddings to a Parquet file."""
    # Read the input CSV file
    df = pl.read_csv(data_path)
    df = df.select(["bio_id", "bio_description"])  # Select only the relevant columns

    # print(df.head());breakpoint()  # Display the first few rows of the DataFrame for verification
    
    # Embed the text values
    texts = df.select("bio_description").to_series().to_list()  # Replace 'text_column' with the actual column name
    embeddings = embed_texts(texts, model=MODEL, batch_size=batch_size, workers=workers)

    print(embeddings.shape)  # Display the shape of the embeddings array for verification
    
    # Save the embeddings to a Parquet file
    embedding_path = embeddings_dir / MODEL.replace("/", "_") / f"{data_path.name.replace('.csv', '')}.parquet"
    # embeddings_dir = Path(embedding_path)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)



    print(f"Saving embeddings to {embedding_path}... Embedding height = {embeddings.shape[0]}")  # Display the path and embedding dimensions for verification
    save_embeddings(
        df.select("bio_id").to_series().to_list(),
        embeddings,
        embedding_path,
    )  # Replace 'id_column' with the actual ID column name
    
    print(f"Embeddings saved to {embedding_path}")




if __name__ == "__main__":
    main()


  