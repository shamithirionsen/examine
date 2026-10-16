"""Embed, cluster and visualise every version found in processed/ (plus raw)."""

import subprocess
from pathlib import Path

ROOT = Path("../data/synthetic_bio")  # ADJUST: where synthetic_bio really is
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def run(*args: str) -> None:
    """Run one command and stop everything if it fails.

    Args:
        *args: The command words.

    Raises:
        subprocess.CalledProcessError: If the command exits with an error.
    """
    print("\n>>>", " ".join(args))
    subprocess.run(args, check=True)


def find_versions(root: Path) -> dict[str, Path]:
    """Map each version name to the folder holding its CSV files.

    Returns:
        "raw" plus one entry per subfolder of processed/ (folders starting
        with "_" are ignored).
    """
    versions = {"raw": root / "raw"}
    for folder in sorted((root / "processed").iterdir()):
        if folder.is_dir() and not folder.name.startswith("_"):
            versions[folder.name] = folder
    return versions

def analyse(root: Path, version: str, text_dir: Path, model: str,
            batch_size: int, workers: int) -> None:
    """Embed, cluster and visualise one version.

    Args:
        version: Version name used to namespace each output stage.
        text_dir: Folder containing the CSV files of this version.
    """
    emb = root / "embeddings" / version
    clu = root / "clusters" / version
    viz = root / "viz" / version
    run("uv", "run", "python", "script/embed.py",
        "--data-path", str(text_dir), "--embeddings_dir", str(emb),
        "--model", model, "--batch_size", str(batch_size),
        "--workers", str(workers))
    run("uv", "run", "python", "script/clusterise.py",
        "--embeddings_path", str(emb), "--clusters_dir", str(clu))
    run("uv", "run", "python", "script/viz.py",
        "--embeddings_path", str(emb), "--data_path", str(text_dir),
        "--clusters_dir", str(clu), "--viz_dir", str(viz), "--version", version)

@click.command()
@click.option("--root", type=click.Path(exists=True, file_okay=False, path_type=Path),
              default=ROOT, show_default=True,
              help="Data root containing raw, processed, and output directories.")
@click.option("--model", default=DEFAULT_MODEL, show_default=True,
              help="Embedding model name passed to embed.py.")
@click.option("--batch-size", default=128, show_default=True, type=int)
@click.option("--workers", default=8, show_default=True, type=int)
@click.option("--version", "version_options", multiple=True,
              help="Version to process; repeat the option for several versions.")
@click.argument("version_args", nargs=-1)
def main(root: Path, model: str, batch_size: int, workers: int,
         version_options: tuple[str, ...], version_args: tuple[str, ...]) -> None:
    """Embed, cluster, and visualise selected versions, or all versions."""
    versions = find_versions(root)
    wanted = version_options or version_args or tuple(versions)
    unknown = sorted(set(wanted) - set(versions))
    if unknown:
        raise click.ClickException(
            f"Unknown version(s): {', '.join(unknown)}. "
            f"Choose from: {', '.join(versions)}"
        )
    for version in wanted:
        analyse(root, version, versions[version], model, batch_size, workers)


if __name__ == "__main__":
    main()