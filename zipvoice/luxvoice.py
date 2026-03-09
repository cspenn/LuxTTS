"""LuxTTS public interface for text-to-speech generation."""

# start zipvoice/luxvoice.py
import structlog
import torch

from zipvoice.config import AudioConfig
from zipvoice.exceptions import AudioProcessingError, ModelLoadError
from zipvoice.generation_types import GenerationConfig, PromptContext
from zipvoice.modeling_utils import generate, load_models, process_audio
from zipvoice.onnx_modeling import generate_cpu

log = structlog.get_logger()

FREQ_RANGE: int = AudioConfig().freq_range
DEFAULT_RMS: float = AudioConfig().default_rms


class LuxTTS:
    """LuxTTS class for encoding prompt and generating speech on cpu/cuda/mps."""

    def __init__(self, model_path: str | None = None, device: str = "auto", threads: int = 4) -> None:
        """Initialize LuxTTS by loading models onto the target device.

        Args:
            model_path: Local directory path for the model. Downloads from
                HuggingFace Hub if None.
            device: Target device. Use ``"auto"`` (default) to select the best
                available device (CUDA → MPS → CPU). Can also be ``"cuda"``,
                ``"mps"``, or ``"cpu"``.
            threads: Number of CPU threads for ONNX inference (CPU mode only).
        """
        # Resolve "auto" and fall back from unavailable devices
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            log.info("device_auto_selected", device=device)
        elif device == "cuda" and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                log.info("cuda_not_available", fallback_device="mps")
                device = "mps"
            else:
                log.info("cuda_not_available", fallback_device="cpu")
                device = "cpu"

        try:
            model, feature_extractor, vocos, tokenizer, transcriber = load_models(
                model_path, device=device, num_thread=threads
            )
            log.info("loading_model", device=device)
        except ModelLoadError:
            raise  # re-raise our own exception type as-is
        except Exception as exc:
            msg = "Failed to initialize LuxTTS model"
            raise ModelLoadError(msg) from exc

        self.model = model
        self.feature_extractor = feature_extractor
        self.vocos = vocos
        self.tokenizer = tokenizer
        self.transcriber = transcriber
        self.device = device
        self.vocos.freq_range = FREQ_RANGE

    def encode_prompt(
        self, prompt_audio: str | None, duration: int = 5, rms: float = DEFAULT_RMS
    ) -> PromptContext | None:
        """Encode an audio prompt for later use in generate_speech().

        Args:
            prompt_audio: Path to the reference audio file.
            duration: Maximum duration in seconds to load from the file.
            rms: Target RMS level for volume normalisation.

        Returns:
            PromptContext with encoded tokens, features, and RMS value.

        Raises:
            AudioProcessingError: If the audio file cannot be processed.
        """
        if prompt_audio is None:
            return None
        try:
            prompt = process_audio(
                prompt_audio,
                self.transcriber,
                self.tokenizer,
                self.feature_extractor,
                self.device,
                target_rms=rms,
                duration=duration,
            )
        except Exception as exc:
            msg = f"Failed to process audio prompt: {prompt_audio}"
            raise AudioProcessingError(msg) from exc
        return prompt

    def generate_speech(  # noqa: PLR0913
        self,
        text: str,
        encode_dict: PromptContext | dict | None,  # type: ignore[type-arg]
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        t_shift: float = 0.5,
        speed: float = 1.0,
        return_smooth: bool = False,
    ) -> torch.Tensor:
        """Encode text and generate speech using the flow-matching model.

        Args:
            text: Text to synthesise.
            encode_dict: PromptContext from encode_prompt(), or legacy dict.
            num_steps: Number of flow-matching diffusion steps.
            guidance_scale: Classifier-free guidance scale.
            t_shift: Time-shift parameter (analogous to temperature).
            speed: Speaking rate multiplier.
            return_smooth: If True, return 24 kHz output; otherwise 48 kHz.

        Returns:
            CPU waveform tensor.
        """
        # Handle both PromptContext and legacy dict for backward compatibility.
        if encode_dict is None:
            msg = "A voice prompt is required for speech generation."
            raise ValueError(msg)
        if isinstance(encode_dict, dict):
            prompt_tokens, prompt_features_lens, prompt_features, prompt_rms = encode_dict.values()
            prompt = PromptContext(
                tokens=prompt_tokens,
                features_lens=prompt_features_lens,
                features=prompt_features,
                rms=prompt_rms,
            )
        else:
            prompt = encode_dict

        config = GenerationConfig(
            num_step=num_steps,
            guidance_scale=guidance_scale,
            speed=speed,
            t_shift=t_shift,
        )

        if return_smooth:
            self.vocos.return_48k = False
        else:
            self.vocos.return_48k = True

        if self.device == "cpu":
            final_wav = generate_cpu(prompt, text, self.model, self.vocos, self.tokenizer, config=config)
        else:
            final_wav = generate(prompt, text, self.model, self.vocos, self.tokenizer, config=config)

        return final_wav.cpu()


# end zipvoice/luxvoice.py
