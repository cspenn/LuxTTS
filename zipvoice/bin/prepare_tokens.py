# start zipvoice/bin/prepare_tokens.py
"""This file reads the texts in given manifest and save the new cuts with prepared tokens."""

from functools import partial
from pathlib import Path

import structlog
import typer
from lhotse import load_manifest, split_parallelize_combine

from zipvoice.tokenizer.tokenizer import add_tokens

log = structlog.get_logger()


app = typer.Typer(help="Read texts from manifest and save cuts with prepared tokens.", add_completion=False)


def prepare_tokens(
    input_file: Path,
    output_file: Path,
    num_jobs: int,
    tokenizer: str,
    lang: str = "en-us",
):
    """Prepare tokens for a manifest file by running the tokenizer.

    Args:
        input_file: Path to the input manifest file.
        output_file: Path to save the output manifest with tokens.
        num_jobs: Number of parallel jobs.
        tokenizer: Tokenizer type name.
        lang: Language identifier for espeak tokenizer.
    """
    log.info("processing", input_file=str(input_file))
    if output_file.is_file():
        log.info("output_exists_skipping", output_file=str(output_file))
        return
    log.info("loading_manifest", input_file=str(input_file))
    cut_set = load_manifest(input_file)

    _add_tokens = partial(add_tokens, tokenizer=tokenizer, lang=lang)

    log.info("adding_tokens")

    cut_set = split_parallelize_combine(num_jobs=num_jobs, manifest=cut_set, fn=_add_tokens)

    log.info("saving_file", output_file=str(output_file))
    cut_set.to_file(output_file)


@app.command()
def main(
    input_file: str = typer.Option(..., "--input-file", help="Input manifest without tokens"),  # noqa: B008
    output_file: str = typer.Option(..., "--output-file", help="Output manifest with tokens."),  # noqa: B008
    num_jobs: int = typer.Option(20, "--num-jobs", help="Number of jobs to run in parallel."),  # noqa: B008
    tokenizer: str = typer.Option(
        "emilia",
        "--tokenizer",
        help="The destination directory of manifest files.",
    ),
    lang: str = typer.Option(
        "en-us",
        "--lang",
        help="Language identifier, used when tokenizer type is espeak. see"
        "https://github.com/rhasspy/espeak-ng/blob/master/docs/languages.md",
    ),
) -> None:
    """Prepare token-augmented manifest files for training."""
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    prepare_tokens(
        input_file=input_path,
        output_file=output_path,
        num_jobs=num_jobs,
        tokenizer=tokenizer,
        lang=lang,
    )

    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/prepare_tokens.py
