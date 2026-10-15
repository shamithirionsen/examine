"""
Reads files from a specified directory, processes the text data to replace emoticons with their textual representation, and saves the processed data to a new directory.
Takes in a string and replaces emoticons with their textual representation. 
For example, ":)" becomes "smile". This is useful for text processing tasks where emoticons may not be handled well by models.
Save as a new column in the dataframe.
Unstick # form text, save as a new column in the dataframe.

"""

from email.mime import text

import click
import polars as pl
from pathlib import Path
import unicodedata
import emoji
import re 
import string



def demojize_text(text: str, lang: str = "fr") -> str:
    """Replace every emoji in `text` with its localized textual name.
    Replaces all punctuations with a space.

    Args:
        text: Input string, possibly containing emojis.
        lang: Language code for emoji names ("en", "fr", ...).

    Returns:
        The text with emojis replaced by `:name:` tokens.
    """
    text = emoji.demojize(text, language=lang)
    text = re.sub(r"[:]", "", text)  # remove colons
    return " ".join(text.replace("_", " ").split())


@click.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default="../data/synthetic_bio/raw",
    help="Path to the folder containing CSV files.",
)
@click.option(
    "--output_dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default="../data/synthetic_bio/processed/emoji_text",
    help="Path to the output folder for the CSV files.",
)
def main(data_dir: Path, output_dir: Path) -> None:
    """Process CSV files in the specified directory to replace  emoticons with 
    their textual representation and save the processed files to a new directory.
    
    
    
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in data_dir.glob("*.csv"):
        df = pl.read_csv(csv_file)
        if "bio_description" not in df.columns:
            raise click.ClickException(f"'bio_description' column not found in {csv_file}")

        # Replace emoticons with their textual representation
        df = df.with_columns(
            pl.col("bio_description").map_elements(
                demojize_text,
                return_dtype=pl.String,
            ).alias("bio_description_demojized") # for parsing column names use startswith("bio_description") 
        )
        df = df.drop("bio_description") # drop or keep ?

        print(f"Head of processed DataFrame for {csv_file.name}:")
        print(df.head()); breakpoint()

        output_file = output_dir / csv_file.name
        df.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")


if __name__ == "__main__":
    main()