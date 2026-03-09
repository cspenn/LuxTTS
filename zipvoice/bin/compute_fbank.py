# start zipvoice/bin/compute_fbank.py
#!/usr/bin/env python3
# Copyright    2024-2025  Xiaomi Corp.        (authors: Wei Kang
#                                                       Han Zhu)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compute fbank features from lhotse manifests.

Usage::

      python3 -m zipvoice.bin.compute_fbank \
        --source-dir data/manifests \
        --dest-dir data/fbank \
        --dataset libritts \
        --subset dev-other \
        --sampling-rate 24000 \
        --num-jobs 20.

The input would be data/manifests/libritts-cuts_dev-other.jsonl.gz or
    (libritts_supervisions_dev-other.jsonl.gz and librittsrecordings_dev-other.jsonl.gz)

The output would be data/fbank/libritts-cuts_dev-other.jsonl.gz
"""

from concurrent.futures import ProcessPoolExecutor as Pool
from pathlib import Path

import lhotse
import structlog
import torch
import typer
from lhotse import CutSet, LilcomChunkyWriter, load_manifest_lazy

from zipvoice.utils.common import AttributeDict
from zipvoice.utils.feature import VocosFbank

log = structlog.get_logger()

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

lhotse.set_audio_duration_mismatch_tolerance(0.1)


app = typer.Typer(help="Compute fbank features from lhotse manifests.", add_completion=False)


def compute_fbank_split_single(params, idx):  # noqa: PLR0913
    """Compute fbank features for a single split of the dataset.

    Args:
        params: Parameters containing configuration for processing.
        idx: Index of the split to process.
    """
    log.info(
        "computing_features_split",
        idx=idx,
        dataset=params.dataset,
        subset=params.subset,
    )
    lhotse.set_audio_duration_mismatch_tolerance(0.1)  # for emilia
    src_dir = Path(params.source_dir)
    output_dir = Path(params.dest_dir)

    if not src_dir.exists():
        log.error("src_dir_not_exists", src_dir=str(src_dir))
        return

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    num_digits = 8
    if params.type == "vocos":
        extractor = VocosFbank()
    else:
        msg = f"{params.type} is not supported"
        raise ValueError(msg)

    prefix = params.dataset
    subset = params.subset
    suffix = "jsonl.gz"

    idx = f"{idx}".zfill(num_digits)
    cuts_filename = f"{prefix}_cuts_{subset}.{idx}.{suffix}"

    if (src_dir / cuts_filename).is_file():
        log.info("loading_manifests", path=str(src_dir / cuts_filename))
        cut_set = load_manifest_lazy(src_dir / cuts_filename)
    else:
        log.warning("cuts_file_not_exists_skipping", cuts_filename=cuts_filename)
        return

    cut_set = cut_set.resample(params.sampling_rate)

    if (output_dir / cuts_filename).is_file():
        log.info("cuts_already_exists_skipping", cuts_filename=cuts_filename)
        return

    log.info("processing_subset", subset=subset, idx=idx, prefix=prefix)

    cut_set = cut_set.compute_and_store_features_batch(
        extractor=extractor,
        storage_path=f"{output_dir}/{prefix}_feats_{subset}_{idx}",
        num_workers=4,
        batch_duration=params.batch_duration,
        storage_type=LilcomChunkyWriter,
        overwrite=True,
    )
    log.info("saving_file", path=str(output_dir / cuts_filename))
    cut_set.to_file(output_dir / cuts_filename)


def compute_fbank_split(params):
    """Compute fbank features for split manifest files in parallel.

    Args:
        params: Parameters containing split configuration.
    """
    if params.split_end < params.split_begin:
        log.warning(
            "split_begin_gt_split_end",
            split_begin=params.split_begin,
            split_end=params.split_end,
        )

    with Pool(max_workers=params.num_jobs) as pool:
        futures = [
            pool.submit(compute_fbank_split_single, params, i) for i in range(params.split_begin, params.split_end)
        ]
        for f in futures:
            f.result()
            f.done()


def compute_fbank(params):
    """Compute fbank features for a complete manifest file.

    Args:
        params: Parameters containing processing configuration.
    """
    log.info(
        "computing_features",
        dataset=params.dataset,
        subset=params.subset,
    )
    src_dir = Path(params.source_dir)
    output_dir = Path(params.dest_dir)
    num_jobs = params.num_jobs
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    prefix = params.dataset
    subset = params.subset
    suffix = "jsonl.gz"

    cut_set_name = f"{prefix}_cuts_{subset}.{suffix}"

    if (src_dir / cut_set_name).is_file():
        log.info("loading_manifests", path=str(src_dir / cut_set_name))
        cut_set = load_manifest_lazy(src_dir / cut_set_name)
    else:
        recordings = load_manifest_lazy(src_dir / f"{prefix}_recordings_{subset}.{suffix}")
        supervisions = load_manifest_lazy(src_dir / f"{prefix}_supervisions_{subset}.{suffix}")
        cut_set = CutSet.from_manifests(
            recordings=recordings,
            supervisions=supervisions,
        )

    cut_set = cut_set.resample(params.sampling_rate)
    if params.type == "vocos":
        extractor = VocosFbank()
    else:
        msg = f"{params.type} is not supported"
        raise ValueError(msg)

    cuts_filename = f"{prefix}_cuts_{subset}.{suffix}"
    if (output_dir / cuts_filename).is_file():
        log.info("already_exists_skipping", prefix=prefix, subset=subset)
        return
    log.info("processing_subset", subset=subset, prefix=prefix)

    cut_set = cut_set.compute_and_store_features(
        extractor=extractor,
        storage_path=f"{output_dir}/{prefix}_feats_{subset}",
        num_jobs=num_jobs,
        storage_type=LilcomChunkyWriter,
    )
    log.info("saving_file", path=str(output_dir / cuts_filename))
    cut_set.to_file(output_dir / cuts_filename)


@app.command()
def main(  # noqa: PLR0913
    sampling_rate: int = typer.Option(  # noqa: B008
        24000,
        "--sampling-rate",
        help="The target sampling rate, the audio will be resampled to it.",
    ),
    type: str = typer.Option("vocos", "--type", help="fbank type"),  # noqa: B008
    dataset: str = typer.Option(..., "--dataset", help="Dataset name."),  # noqa: B008
    subset: str = typer.Option(..., "--subset", help="The subset of the dataset."),  # noqa: B008
    source_dir: str = typer.Option(  # noqa: B008
        "data/manifests",
        "--source-dir",
        help="The source directory of manifest files.",
    ),
    dest_dir: str = typer.Option(  # noqa: B008
        "data/fbank",
        "--dest-dir",
        help="The destination directory of manifest files.",
    ),
    split_cuts: bool = typer.Option(  # noqa: B008
        False, "--split-cuts", help="Whether to use splited cuts."
    ),
    split_begin: int = typer.Option(  # noqa: B008
        None, "--split-begin", help="Start idx of splited cuts."
    ),
    split_end: int = typer.Option(  # noqa: B008
        None, "--split-end", help="End idx of splited cuts."
    ),
    batch_duration: int = typer.Option(  # noqa: B008
        1000,
        "--batch-duration",
        help="The batch duration when computing the features.",
    ),
    num_jobs: int = typer.Option(  # noqa: B008
        20, "--num-jobs", help="The number of extractor workers."
    ),
) -> None:
    """Compute fbank features and save to disk."""
    params = AttributeDict(
        sampling_rate=sampling_rate,
        type=type,
        dataset=dataset,
        subset=subset,
        source_dir=source_dir,
        dest_dir=dest_dir,
        split_cuts=split_cuts,
        split_begin=split_begin,
        split_end=split_end,
        batch_duration=batch_duration,
        num_jobs=num_jobs,
    )
    log.info("params", **dict(params))
    if split_cuts:
        compute_fbank_split(params=params)
    else:
        compute_fbank(params=params)
    log.info("done")


if __name__ == "__main__":
    app()

# end zipvoice/bin/compute_fbank.py
