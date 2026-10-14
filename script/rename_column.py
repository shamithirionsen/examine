"""
Rename selected columns to a standardized name for consistency across different data sources.

"""



import click
import polars as pl
from pathlib import Path


@click.command()
@click.option("--data_path", type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path), 
              default=Path("../data/synthetic_bio/raw/synthetic_twitter_bios_200.csv"), help="Path to the input CSV file.")
@click.option("--output_path", type=click.Path(path_type=Path),
              default=Path("../data/synthetic_bio/raw/synthetic_twitter_bios_200_renamed.csv"), help="Path to the output CSV file with renamed columns.")


def main(data_path: Path, output_path: Path) -> None:
    """Rename selected columns in the input CSV file and save the result to a new CSV file."""
    # Read the input CSV file

    print(f"Reading data from {data_path}...")
    df = pl.read_csv(data_path)

    # display column headers for verification
    print(f"Original columns: {df.columns}")

    # Define a mapping of old column names to new standardized names
    rename_mapping = {
        "id": "bio_id",
        "authorbioDescription": "bio_description",
    }

    # Rename the columns using the mapping
    df_renamed = df.rename(rename_mapping)

    # Save the DataFrame with renamed columns to a new CSV file
    df_renamed.write_csv(output_path)
    print(f"Renamed columns and saved to {output_path}")



if __name__ == "__main__":
    main()  