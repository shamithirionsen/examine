"""


"""


import click
import polars as pl
from pathlib import Path
import regex  # pip install regex; supports \p{Lu} / \p{Ll}

HASHTAG = regex.compile(r"#(\w+)")
CAMEL_LOWER_UPPER = regex.compile(r"(?<=[\p{Ll}\d])(?=\p{Lu})") # lower/digit -> Upper boundary, e.g. "StandWith" -> "Stand With"
CAMEL_ACRONYM = regex.compile(r"(?<=\p{Lu})(?=\p{Lu}\p{Ll})") # acronym -> Word boundary, e.g. "USAToday" -> "USA Today"




def separate_hashtag(text: str) -> str:
    """Put a space after each '#' so it becomes its own token.

    Args:
        text: Input text containing hashtags.

    Returns:
        Text where '#StandWithIsrael' becomes '# StandWithIsrael'.
    """
    return HASHTAG.sub(r"# \1", text)

def split_hashtag_camel(text: str) -> str:
    """Separate '#' and split the hashtag body on capital letters.

    Args:
        text: Input text containing hashtags.

    Returns:
        Text where '#StandWithIsrael' becomes '# Stand With Israel'.
    """

    def _split(match: regex.Match[str]) -> str:
        body = CAMEL_LOWER_UPPER.sub(" ", match.group(1))
        body = CAMEL_ACRONYM.sub(" ", body)
        return f"# {body}"

    return HASHTAG.sub(_split, text)


@click.command()
@click.option("--data-dir", default="../data/synthetic_bio/raw", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-dir-hashtag", default="../data/synthetic_bio/processed/hashtag_sep", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir-camel", default="../data/synthetic_bio/processed/hashtag_camel", type=click.Path(path_type=Path), required=True)
def main(data_dir: Path, output_dir_hashtag: Path, output_dir_camel: Path) -> None:
    """Process CSV files in the specified directory to separate hashtags into a new column and save the processed files to a new directory.

    Args:
        data_dir: Path to the folder containing CSV files.
        output_dir: Path to the output folder for the CSV files.  
    Returns:
        None. Processed CSV files are written to the output directory.
    """
    output_dir_hashtag.mkdir(parents=True, exist_ok=True)
    output_dir_camel.mkdir(parents=True, exist_ok=True)

    for csv_file in data_dir.rglob("*.csv"):
        df = pl.read_csv(csv_file)
        if "bio_description" not in df.columns:
            raise click.ClickException(f"'bio_description' column not found in {csv_file}")
        df = df.with_columns(
            pl.col("bio_description").map_elements(
                separate_hashtag, return_dtype=pl.Utf8
            ).alias("bio_description_hashtag_sep")
        )
        output_file = output_dir_hashtag / csv_file.name
        df = df.with_columns(
            pl.col("bio_description").map_elements(
                split_hashtag_camel, return_dtype=pl.Utf8
            ).alias("bio_description_hashtag_uppercase_split")
        )
        output_file = output_dir_camel / csv_file.name
        df.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")



if __name__ == "__main__":
    main()  