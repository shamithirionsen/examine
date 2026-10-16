"""


"""


import click
import polars as pl
from pathlib import Path
import regex  # pip install regex; supports \p{Lu} / \p{Ll}
from textualise_emoticons import demojize_text

HASHTAG = regex.compile(r"#\s*(\w+)")
CAMEL_LOWER_UPPER = regex.compile(r"(?<=[\p{Ll}\d])(?=\p{Lu})") # lower/digit -> Upper boundary, e.g. "StandWith" -> "Stand With"
CAMEL_ACRONYM = regex.compile(r"(?<=\p{Lu})(?=\p{Lu}\p{Ll})") # acronym -> Word boundary, e.g. "USAToday" -> "USA Today"




def split_hashtag(text: str) -> str:
    """Put a space after each '#' so it becomes its own token.

    Args:
        text: Input text containing hashtags.

    Returns:
        Text where '#StandWithIsrael' becomes '# StandWithIsrael'.
    """
    return HASHTAG.sub(r"# \1", text)

def split_camel_case(text: str) -> str:
    """Split camel case words in the text.

    Args:
        text: Input text containing camel case words.

    Returns:
        Text where 'StandWithIsrael' becomes 'Stand With Israel'.
    """
    text = CAMEL_LOWER_UPPER.sub(" ", text)
    text = CAMEL_ACRONYM.sub(" ", text)
    return text

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

def remove_hashtags(text: str) -> str:
    """Remove hashtags from the text.

    Args:
        text: Input text containing hashtags.
    Returns:
        Text with hashtags removed.
    """
    # print(
    #     f"Removing hashtags from: {text} check: {text.replace('#', '').strip()}",
    #     flush=True,
    # )
    return text.replace("#", "").strip()  # Remove '#' and strip whitespace


@click.command()
@click.option("--data-dir", default="../data/synthetic_bio/raw", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-dir-hashtag", default="../data/synthetic_bio/processed/hashtag_sep", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir-camel", default="../data/synthetic_bio/processed/camel", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir-hashtag-camel", default="../data/synthetic_bio/processed/hashtag_camel", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir-hashtag-camel-demotized", default="../data/synthetic_bio/processed/hashtag_camel_demotized", type=click.Path(path_type=Path), required=True)
@click.option("--output-dir-no-hashtags", default="../data/synthetic_bio/processed/no_hashtags", type=click.Path(path_type=Path), required=True)

def main(data_dir: Path, output_dir_hashtag: Path, output_dir_camel: Path, 
         output_dir_hashtag_camel: Path, output_dir_hashtag_camel_demotized: Path,
         output_dir_no_hashtags: Path) -> None:
    """Process CSV files in the specified directory to separate hashtags into a new column and save the processed files to a new directory.

    Args:
        data_dir: Path to the folder containing CSV files.
        output_dir_hashtag: Path to the output folder for the hashtag-separated CSV files.
        output_dir_camel: Path to the output folder for the camel-case-separated CSV files.
        output_dir_hashtag_camel: Path to the output folder for the hashtag and camel-case-separated CSV files.
        output_dir_hashtag_camel_demotized: Path to the output folder for the hashtag and camel-case-separated CSV files with emojis demotized.
    Returns:
        None. Processed CSV files are written to the output directory.
    """
    output_dir_hashtag.mkdir(parents=True, exist_ok=True)
    output_dir_camel.mkdir(parents=True, exist_ok=True)
    output_dir_hashtag_camel.mkdir(parents=True, exist_ok=True)
    output_dir_hashtag_camel_demotized.mkdir(parents=True, exist_ok=True)
    output_dir_no_hashtags.mkdir(parents=True, exist_ok=True)

    for csv_file in data_dir.rglob("*.csv"):
        df = pl.read_csv(csv_file)
        if "bio_description" not in df.columns:
            raise click.ClickException(f"'bio_description' column not found in {csv_file}")
        df_hastag_sep = df.with_columns(
            pl.col("bio_description").map_elements(
                split_hashtag, return_dtype=pl.Utf8
            ).alias("bio_description_hashtag_split") # starts with "bio_description"
        )
        df_hashtag_sep = df_hastag_sep.drop("bio_description") # drop or keep ?
        
        output_file = output_dir_hashtag / csv_file.name
        df_hastag_sep.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")

        df_camel = df.with_columns(
            pl.col("bio_description").map_elements(
                split_camel_case, return_dtype=pl.Utf8
            ).alias("bio_description_camel_split")
        )
        df_camel = df_camel.drop("bio_description") # drop or keep ?
        output_file = output_dir_camel / csv_file.name
        df_camel.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")


        # hastag and camel sep
        df_hashtag_camel = df.with_columns(
            pl.col("bio_description").map_elements(
                split_hashtag_camel, return_dtype=pl.Utf8
            ).alias("bio_description_hashtag_camel_split")
        )
        df_hashtag_camel = df_hashtag_camel.drop("bio_description") # drop or keep ?
        output_file = output_dir_hashtag_camel / csv_file.name
        df_hashtag_camel.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")

        ### Check from here:
        # hastag and camel sep with emojis demotized
        df_hashtag_camel_demotized = df_hashtag_camel.with_columns(
            pl.col("bio_description_hashtag_camel_split").map_elements(
                lambda x: demojize_text(x), return_dtype=pl.Utf8
            ).alias("bio_description_hashtag_camel_demotized")
        )

        
        df_hashtag_camel_demotized = df_hashtag_camel_demotized.drop("bio_description_hashtag_camel_split") # drop or keep ?
        output_file = output_dir_hashtag_camel_demotized / csv_file.name
        df_hashtag_camel_demotized.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")


        # remove hastags from final version
        df_no_hashtags = df_hashtag_camel_demotized.with_columns(
            pl.col("bio_description_hashtag_camel_demotized").map_elements(
                remove_hashtags, return_dtype=pl.Utf8
            ).alias("bio_description_no_hashtags")
        )
        df_no_hashtags = df_no_hashtags.drop("bio_description_hashtag_camel_demotized") # drop or keep ?
        output_file = output_dir_no_hashtags / csv_file.name
        df_no_hashtags.write_csv(output_file)
        click.echo(f"Processed file saved to {output_file}")





if __name__ == "__main__":
    main()  