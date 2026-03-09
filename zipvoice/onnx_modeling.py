"""ONNX-based inference models for CPU generation."""

# start zipvoice/onnx_modeling.py
from typing import Any

import onnxruntime as ort
import structlog
import torch
from torch import Tensor

from zipvoice.generation_types import GenerationConfig, PromptContext
from zipvoice.models.modules.solver import get_time_steps

log = structlog.get_logger()

DEFAULT_FEAT_SCALE = 0.1
DEFAULT_SPEED_FACTOR = 1.3


class OnnxModel:
    """ONNX runtime wrapper for the LuxTTS text encoder and flow-matching decoder."""

    def __init__(
        self,
        text_encoder_path: str,
        fm_decoder_path: str,
        num_thread: int = 1,
    ):
        """Initialize OnnxModel by loading ONNX sessions for text encoder and FM decoder.

        Args:
            text_encoder_path: Filesystem path to the text encoder ONNX model.
            fm_decoder_path: Filesystem path to the flow-matching decoder ONNX model.
            num_thread: Number of inter/intra-op threads for ONNX inference.
        """
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = num_thread
        session_opts.intra_op_num_threads = num_thread

        self.session_opts = session_opts

        self.init_text_encoder(text_encoder_path)
        self.init_fm_decoder(fm_decoder_path)

    def init_text_encoder(self, model_path: str) -> None:
        """Load the text encoder ONNX session.

        Args:
            model_path: Filesystem path to the text encoder ONNX file.
        """
        self.text_encoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )

    def init_fm_decoder(self, model_path: str) -> None:
        """Load the flow-matching decoder ONNX session and read model metadata.

        Args:
            model_path: Filesystem path to the FM decoder ONNX file.
        """
        self.fm_decoder = ort.InferenceSession(
            model_path,
            sess_options=self.session_opts,
            providers=["CPUExecutionProvider"],
        )
        meta = self.fm_decoder.get_modelmeta().custom_metadata_map
        self.feat_dim = int(meta["feat_dim"])

    def run_text_encoder(
        self,
        tokens: Tensor,
        prompt_tokens: Tensor,
        prompt_features_len: Tensor,
        speed: Tensor,
    ) -> Tensor:
        """Run the text encoder and return the conditioning tensor.

        Args:
            tokens: Token IDs for the target text, shape (1, T).
            prompt_tokens: Token IDs for the voice prompt, shape (1, P).
            prompt_features_len: Scalar tensor with the prompt feature length.
            speed: Scalar tensor with the speaking-rate multiplier.

        Returns:
            Text conditioning tensor of shape (batch, frames, dim).
        """
        out = self.text_encoder.run(
            [
                self.text_encoder.get_outputs()[0].name,
            ],
            {
                self.text_encoder.get_inputs()[0].name: tokens.numpy(),
                self.text_encoder.get_inputs()[1].name: prompt_tokens.numpy(),
                self.text_encoder.get_inputs()[2].name: prompt_features_len.numpy(),
                self.text_encoder.get_inputs()[3].name: speed.numpy(),
            },
        )
        return torch.from_numpy(out[0])

    def run_fm_decoder(
        self,
        t: Tensor,
        x: Tensor,
        text_condition: Tensor,
        speech_condition: torch.Tensor,
        guidance_scale: Tensor,
    ) -> Tensor:
        """Run one flow-matching decoder step and return the predicted velocity.

        Args:
            t: Scalar timestep tensor.
            x: Noisy latent tensor of shape (batch, frames, feat_dim).
            text_condition: Text conditioning tensor from the text encoder.
            speech_condition: Speech conditioning tensor (padded prompt features).
            guidance_scale: Classifier-free guidance scale as a scalar tensor.

        Returns:
            Predicted velocity tensor of shape (batch, frames, feat_dim).
        """
        out = self.fm_decoder.run(
            [
                self.fm_decoder.get_outputs()[0].name,
            ],
            {
                self.fm_decoder.get_inputs()[0].name: t.numpy(),
                self.fm_decoder.get_inputs()[1].name: x.numpy(),
                self.fm_decoder.get_inputs()[2].name: text_condition.numpy(),
                self.fm_decoder.get_inputs()[3].name: speech_condition.numpy(),
                self.fm_decoder.get_inputs()[4].name: guidance_scale.numpy(),
            },
        )
        return torch.from_numpy(out[0])


def sample(  # noqa: PLR0913
    model: OnnxModel,
    tokens: list[list[int]],
    prompt_tokens: list[list[int]],
    prompt_features: Tensor,
    speed: float = DEFAULT_SPEED_FACTOR,
    t_shift: float = 0.5,
    guidance_scale: float = 1.0,
    num_step: int = 16,
) -> torch.Tensor:
    """Run the ONNX flow-matching sampling loop and return the predicted features.

    Args:
        model: OnnxModel wrapping the text encoder and flow-matching decoder.
        tokens: Nested list of token IDs for the target text (batch size must be 1).
        prompt_tokens: Nested list of token IDs for the voice prompt (batch size must be 1).
        prompt_features: Prompt filterbank features, shape (1, frames, feat_dim).
        speed: Speaking-rate multiplier applied inside the text encoder.
        t_shift: Time-shift parameter for the ODE schedule.
        guidance_scale: Classifier-free guidance scale.
        num_step: Number of flow-matching ODE steps.

    Returns:
        Predicted feature tensor of shape (1, generated_frames, feat_dim).
    """
    # --- Preparation ---
    if len(tokens) != 1 or len(prompt_tokens) != 1:
        msg = f"Expected len(tokens)==1 and len(prompt_tokens)==1, got {len(tokens)} and {len(prompt_tokens)}"
        raise ValueError(msg)
    tokens_t = torch.tensor(tokens, dtype=torch.int64)
    prompt_tokens_t = torch.tensor(prompt_tokens, dtype=torch.int64)
    prompt_features_len_t = torch.tensor(prompt_features.size(1), dtype=torch.int64)
    speed_t = torch.tensor(speed, dtype=torch.float32)

    # Run text encoder

    text_condition = model.run_text_encoder(tokens_t, prompt_tokens_t, prompt_features_len_t, speed_t)
    batch_size, num_frames, _ = text_condition.shape
    feat_dim = model.feat_dim

    # Get the time schedule
    timesteps = get_time_steps(
        t_start=0.0,
        t_end=1.0,
        num_step=num_step,
        t_shift=t_shift,
    )

    # Initialize x with noise (x_0)
    x = torch.randn(batch_size, num_frames, feat_dim)
    speech_condition = torch.nn.functional.pad(prompt_features, (0, 0, 0, num_frames - prompt_features.shape[1]))
    guidance_scale_t = torch.tensor(guidance_scale, dtype=torch.float32)

    # --- Sampling Loop ---
    for step in range(num_step):
        t_cur = timesteps[step]
        t_next = timesteps[step + 1]

        # Predict velocity v
        v = model.run_fm_decoder(
            t=t_cur,
            x=x,
            text_condition=text_condition,
            speech_condition=speech_condition,
            guidance_scale=guidance_scale_t,
        )

        # Flow matching formula: x_t = (1 - t) * x_0 + t * x_1
        # Therefore: v = x_1 - x_0
        # This implies:
        x_1_pred = x + (1.0 - t_cur) * v
        x_0_pred = x - t_cur * v

        x = (1.0 - t_next) * x_0_pred + t_next * x_1_pred if step < num_step - 1 else x_1_pred

    # Remove the prompt portion from the generated sequence
    x = x[:, int(prompt_features_len_t.item()) :, :]
    return x


def generate_cpu(  # noqa: PLR0913
    prompt: PromptContext,
    text: str,
    model: Any,
    vocoder: Any,
    tokenizer: Any,
    config: GenerationConfig | None = None,
) -> torch.Tensor:
    """Generate speech for the given text on CPU using ONNX models.

    Args:
        prompt: Encoded voice prompt produced by process_audio().
        text: Text to synthesise.
        model: OnnxModel wrapping the text encoder and flow-matching decoder.
        vocoder: Vocos vocoder for converting features to waveform.
        tokenizer: Tokenizer for encoding the input text.
        config: Generation hyperparameters; defaults to GenerationConfig().

    Returns:
        Generated waveform tensor of shape (1, samples).
    """
    if config is None:
        config = GenerationConfig()

    tokens = tokenizer.texts_to_token_ids([text])
    speed = config.speed * DEFAULT_SPEED_FACTOR  # default is too slow

    pred_features = sample(
        model=model,
        tokens=tokens,
        prompt_tokens=prompt.tokens,
        prompt_features=prompt.features,
        speed=speed,
        t_shift=config.t_shift,
        guidance_scale=config.guidance_scale,
        num_step=config.num_step,
    )

    # Convert to waveform
    pred_features = pred_features.permute(0, 2, 1) / DEFAULT_FEAT_SCALE
    wav: torch.Tensor = vocoder.decode(pred_features).squeeze(1).clamp(-1, 1)

    # Volume matching
    if prompt.rms < config.target_rms:
        wav = wav * (prompt.rms / config.target_rms)

    return wav


# end zipvoice/onnx_modeling.py
