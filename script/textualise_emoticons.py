"""
Takes in a string and replaces emoticons with their textual representation. 
For example, ":)" becomes "smile". This is useful for text processing tasks where emoticons may not be handled well by models.
Save as a new column in the dataframe.
Unstick # form text, save as a new column in the dataframe.

"""

import click
import polars as pl


@click.command()


