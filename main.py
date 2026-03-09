# start main.py
"""LuxTTS entry point.

Orchestrates the LuxTTS text-to-speech pipeline. Contains no core logic —
all generation is delegated to zipvoice.luxvoice.LuxTTS.

Usage:
    python -m zipvoice.main --text "Hello world" --output output.wav
    python main.py --text "Hello world" --output output.wav
"""
from pathlib import Path
from typing import Annotated

import structlog
import typer

from zipvoice.modeling_utils import SAMPLE_RATE_OUTPUT

log = structlog.get_logger()


def main(
    text: Annotated[str, typer.Option(help="Text to synthesize")],
    output: Annotated[str, typer.Option(help="Output .wav file path")],
    prompt: Annotated[str | None, typer.Option(help="Reference audio file for voice cloning")] = None,
    device: Annotated[str, typer.Option(help="Device: cuda, mps, cpu, or auto")] = "auto",
    model_path: Annotated[str | None, typer.Option(help="Local model path (default: download from HuggingFace)")] = None,
    steps: Annotated[int, typer.Option(help="Number of diffusion steps")] = 4,
    speed: Annotated[float, typer.Option(help="Speech speed multiplier")] = 1.0,
) -> None:
    """Run LuxTTS text-to-speech from the command line.

    Parses arguments, initializes the model, and generates speech.
    All core logic is in zipvoice.luxvoice.LuxTTS.
    """
    from zipvoice.luxvoice import LuxTTS

    tts = LuxTTS(model_path=model_path, device=device)
    log.info("model_initialized")

    if prompt:
        log.info("encoding_voice_prompt", prompt=prompt)
        encode_dict = tts.encode_prompt(prompt)
    else:
        encode_dict = tts.encode_prompt(None)

    log.info("generating_speech", text_preview=text[:60])
    audio = tts.generate_speech(text, encode_dict, num_steps=steps, speed=speed)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import torchaudio

    torchaudio.save(str(output_path), audio.cpu(), SAMPLE_RATE_OUTPUT)
    log.info("output_saved", path=str(output_path))


if __name__ == "__main__":
    typer.run(main)
# end main.py
