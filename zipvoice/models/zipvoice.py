"""ZipVoice: flow-matching TTS model with text and speech conditioning."""

# start zipvoice/models/zipvoice.py
# Copyright    2024    Xiaomi Corp.        (authors:  Wei Kang
#                                                     Han Zhu)
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

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from zipvoice.models.modules.solver import EulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.utils.common import (
    condition_time_mask,
    get_tokens_index,
    make_pad_mask,
    pad_labels,
    prepare_avg_tokens_durations,
)


class ZipVoice(nn.Module):
    """The ZipVoice model."""

    def __init__(  # noqa: PLR0913
        self,
        fm_decoder_downsampling_factor: list[int] = None,
        fm_decoder_num_layers: list[int] = None,
        fm_decoder_cnn_module_kernel: list[int] = None,
        fm_decoder_feedforward_dim: int = 1536,
        fm_decoder_num_heads: int = 4,
        fm_decoder_dim: int = 512,
        text_encoder_num_layers: int = 4,
        text_encoder_feedforward_dim: int = 512,
        text_encoder_cnn_module_kernel: int = 9,
        text_encoder_num_heads: int = 4,
        text_encoder_dim: int = 192,
        time_embed_dim: int = 192,
        text_embed_dim: int = 192,
        query_head_dim: int = 32,
        value_head_dim: int = 12,
        pos_head_dim: int = 4,
        pos_dim: int = 48,
        feat_dim: int = 100,
        vocab_size: int = 26,
        pad_id: int = 0,
    ):
        """Initialize the model with specified configuration parameters.

        Args:
            fm_decoder_downsampling_factor: List of downsampling factors for each layer
                in the flow-matching decoder.
            fm_decoder_num_layers: List of the number of layers for each block in the
                flow-matching decoder.
            fm_decoder_cnn_module_kernel: List of kernel sizes for CNN modules in the
                flow-matching decoder.
            fm_decoder_feedforward_dim: Dimension of the feedforward network in the
                flow-matching decoder.
            fm_decoder_num_heads: Number of attention heads in the flow-matching
                decoder.
            fm_decoder_dim: Hidden dimension of the flow-matching decoder.
            text_encoder_num_layers: Number of layers in the text encoder.
            text_encoder_feedforward_dim: Dimension of the feedforward network in the
                text encoder.
            text_encoder_cnn_module_kernel: Kernel size for the CNN module in the
                text encoder.
            text_encoder_num_heads: Number of attention heads in the text encoder.
            text_encoder_dim: Hidden dimension of the text encoder.
            time_embed_dim: Dimension of the time embedding.
            text_embed_dim: Dimension of the text embedding.
            query_head_dim: Dimension of the query attention head.
            value_head_dim: Dimension of the value attention head.
            pos_head_dim: Dimension of the position attention head.
            pos_dim: Dimension of the positional encoding.
            feat_dim: Dimension of the acoustic features.
            vocab_size: Size of the vocabulary.
            pad_id: ID used for padding tokens.
        """
        if fm_decoder_cnn_module_kernel is None:
            fm_decoder_cnn_module_kernel = [31, 15, 7, 15, 31]
        if fm_decoder_num_layers is None:
            fm_decoder_num_layers = [2, 2, 4, 4, 4]
        if fm_decoder_downsampling_factor is None:
            fm_decoder_downsampling_factor = [1, 2, 4, 2, 1]
        super().__init__()

        self.fm_decoder = TTSZipformer(
            in_dim=feat_dim * 3,
            out_dim=feat_dim,
            downsampling_factor=fm_decoder_downsampling_factor,
            num_encoder_layers=fm_decoder_num_layers,
            cnn_module_kernel=fm_decoder_cnn_module_kernel,
            encoder_dim=fm_decoder_dim,
            feedforward_dim=fm_decoder_feedforward_dim,
            num_heads=fm_decoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=True,
            time_embed_dim=time_embed_dim,
        )

        self.text_encoder = TTSZipformer(
            in_dim=text_embed_dim,
            out_dim=feat_dim,
            downsampling_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            cnn_module_kernel=text_encoder_cnn_module_kernel,
            encoder_dim=text_encoder_dim,
            feedforward_dim=text_encoder_feedforward_dim,
            num_heads=text_encoder_num_heads,
            query_head_dim=query_head_dim,
            pos_head_dim=pos_head_dim,
            value_head_dim=value_head_dim,
            pos_dim=pos_dim,
            use_time_embed=False,
        )

        self.feat_dim = feat_dim
        self.text_embed_dim = text_embed_dim
        self.pad_id = pad_id

        self.embed = nn.Embedding(vocab_size, text_embed_dim)
        self.solver = EulerSolver(self, func_name="forward_fm_decoder")

    def forward_fm_decoder(  # noqa: PLR0913
        self,
        t: torch.Tensor,
        xt: torch.Tensor,
        text_condition: torch.Tensor,
        speech_condition: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        guidance_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute velocity.

        Args:
            t:  A tensor of shape (N, 1, 1) or a tensor of a float,
                in the range of (0, 1).
            xt: the input of the current timestep, including condition
                embeddings and noisy acoustic features.
            text_condition: the text condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            speech_condition: the speech condition embeddings, with the
                shape (batch, seq_len, emb_dim).
            padding_mask: The mask for padding, True means masked
                position, with the shape (N, T).
            guidance_scale: The guidance scale in classifier-free guidance,
                which is a tensor of shape (N, 1, 1) or a tensor of a float.

        Returns:
            predicted velocity, with the shape (batch, seq_len, emb_dim).
        """
        xt = torch.cat([xt, text_condition, speech_condition], dim=2)

        if t.dim() not in (0, 3):
            msg = f"t must have 0 or 3 dimensions, got {t.dim()}"
            raise ValueError(msg)
        # Handle t with the shape (N, 1, 1):
        # squeeze the last dimension if it's size is 1.
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        # Handle t with a single value: expand to the size of batch size.
        if t.dim() == 0:
            t = t.repeat(xt.shape[0])

        if guidance_scale is not None:
            while guidance_scale.dim() > 1 and guidance_scale.size(-1) == 1:
                guidance_scale = guidance_scale.squeeze(-1)
            if guidance_scale.dim() == 0:
                guidance_scale = guidance_scale.repeat(xt.shape[0])

            vt = self.fm_decoder(x=xt, t=t, padding_mask=padding_mask, guidance_scale=guidance_scale)
        else:
            vt = self.fm_decoder(x=xt, t=t, padding_mask=padding_mask)
        return vt

    def forward_text_embed(
        self,
        tokens: list[list[int]],
    ):
        """Get the text embeddings.

        Args:
            tokens: a list of list of token ids.

        Returns:
            embed: the text embeddings, shape (batch, seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
        """
        device = self.device if isinstance(self, DDP) else next(self.parameters()).device
        tokens_padded = pad_labels(tokens, pad_id=self.pad_id, device=device)  # (B, S)
        embed = self.embed(tokens_padded)  # (B, S, C)
        tokens_lens = torch.tensor([len(token) for token in tokens], dtype=torch.int64, device=device)
        tokens_padding_mask = make_pad_mask(tokens_lens, embed.shape[1])  # (B, S)

        embed = self.text_encoder(x=embed, t=None, padding_mask=tokens_padding_mask)  # (B, S, C)
        return embed, tokens_lens

    def forward_text_condition(
        self,
        embed: torch.Tensor,
        tokens_lens: torch.Tensor,
        features_lens: torch.Tensor,
    ):
        """Get the text condition with the same length of the acoustic feature.

        Args:
            embed: the text embeddings, shape (batch, token_seq_len, emb_dim).
            tokens_lens: the length of each token sequence, shape (batch,).
            features_lens: the length of each acoustic feature sequence,
                shape (batch,).

        Returns:
            text_condition: the text condition, shape
                (batch, feature_seq_len, emb_dim).
            padding_mask: the padding mask of text condition, shape
                (batch, feature_seq_len).
        """
        num_frames = int(features_lens.max())

        padding_mask = make_pad_mask(features_lens, max_len=num_frames)  # (B, T)

        tokens_durations = prepare_avg_tokens_durations(features_lens, tokens_lens)

        tokens_index = get_tokens_index(tokens_durations, num_frames).to(embed.device)  # (B, T)

        text_condition = torch.gather(
            embed,
            dim=1,
            index=tokens_index.unsqueeze(-1).expand(embed.size(0), num_frames, embed.size(-1)),
        )  # (B, T, F)
        return text_condition, padding_mask

    def forward_text_train(
        self,
        tokens: list[list[int]],
        features_lens: torch.Tensor,
    ):
        """Compute text conditioning tensors for training.

        Args:
            tokens: Text token ID sequences, one list per batch item.
            features_lens: Ground-truth acoustic feature lengths, shape (B,).

        Returns:
            A tuple ``(text_condition, padding_mask)`` where
            ``text_condition`` has shape (B, T, feat_dim) and
            ``padding_mask`` has shape (B, T) with ``True`` at padded frames.
        """
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(embed, tokens_lens, features_lens)
        return (
            text_condition,
            padding_mask,
        )

    def forward_text_inference_gt_duration(
        self,
        tokens: list[list[int]],
        features_lens: torch.Tensor,
        prompt_tokens: list[list[int]],
        prompt_features_lens: torch.Tensor,
    ):
        """Compute text conditioning for inference using ground-truth feature lengths.

        Prepends prompt tokens to each text token sequence and computes
        conditioning aligned to the combined (prompt + target) frame length.

        Args:
            tokens: Target text token ID sequences, one list per batch item.
            features_lens: Ground-truth target feature lengths, shape (B,).
            prompt_tokens: Prompt transcription token ID sequences.
            prompt_features_lens: Prompt feature lengths, shape (B,).

        Returns:
            A tuple ``(text_condition, padding_mask)`` aligned to the combined
            feature length, where ``text_condition`` has shape (B, T, feat_dim)
            and ``padding_mask`` has shape (B, T).
        """
        tokens = [prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens, strict=False)]
        features_lens = prompt_features_lens + features_lens
        embed, tokens_lens = self.forward_text_embed(tokens)
        text_condition, padding_mask = self.forward_text_condition(embed, tokens_lens, features_lens)
        return text_condition, padding_mask

    def forward_text_inference_ratio_duration(
        self,
        tokens: list[list[int]],
        prompt_tokens: list[list[int]],
        prompt_features_lens: torch.Tensor,
        speed: float,
    ):
        """Compute text conditioning for inference with duration predicted from token ratio.

        Target feature length is estimated as ``prompt_frames * (target_tokens
        / prompt_tokens) / speed``, removing the need for a separate duration
        model.

        Args:
            tokens: Target text token ID sequences, one list per batch item.
            prompt_tokens: Prompt transcription token ID sequences.
            prompt_features_lens: Prompt feature lengths, shape (B,).
            speed: Speaking rate multiplier; values >1 produce faster speech.

        Returns:
            A tuple ``(text_condition, padding_mask)`` aligned to the predicted
            combined feature length, where ``text_condition`` has shape
            (B, T, feat_dim) and ``padding_mask`` has shape (B, T).
        """
        device = self.device if isinstance(self, DDP) else next(self.parameters()).device

        cat_tokens = [prompt_token + token for prompt_token, token in zip(prompt_tokens, tokens, strict=False)]

        prompt_tokens_lens = torch.tensor(
            [len(token) for token in prompt_tokens],
            dtype=torch.int64,
            device=device,
        )

        tokens_lens = torch.tensor(
            [len(token) for token in tokens],
            dtype=torch.int64,
            device=device,
        )

        cat_embed, cat_tokens_lens = self.forward_text_embed(cat_tokens)

        features_lens = prompt_features_lens + torch.ceil(
            prompt_features_lens / prompt_tokens_lens * tokens_lens / speed
        ).to(dtype=torch.int64)

        text_condition, padding_mask = self.forward_text_condition(cat_embed, cat_tokens_lens, features_lens)
        return text_condition, padding_mask

    def forward(  # noqa: PLR0913
        self,
        tokens: list[list[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        condition_drop_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Forward pass of the model for training.

        Args:
            tokens: a list of list of token ids.
            features: the acoustic features, with the shape (batch, seq_len, feat_dim).
            features_lens: the length of each acoustic feature sequence, shape (batch,).
            noise: the intitial noise, with the shape (batch, seq_len, feat_dim).
            t: the time step, with the shape (batch, 1, 1).
            condition_drop_ratio: the ratio of dropped text condition.

        Returns:
            fm_loss: the flow-matching loss.
        """
        (
            text_condition,
            padding_mask,
        ) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition_mask = condition_time_mask(
            features_lens=features_lens,
            mask_percent=(0.7, 1.0),
            max_len=features.size(1),
        )
        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        if condition_drop_ratio > 0.0:
            drop_mask = torch.rand(text_condition.size(0), 1, 1).to(text_condition.device) > condition_drop_ratio
            text_condition = text_condition * drop_mask

        xt = features * t + noise * (1 - t)
        ut = features - noise  # (B, T, F)

        vt = self.forward_fm_decoder(
            t=t,
            xt=xt,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
        )

        loss_mask = speech_condition_mask & (~padding_mask)
        fm_loss = torch.mean((vt[loss_mask] - ut[loss_mask]) ** 2)

        return fm_loss

    def sample(  # noqa: PLR0913
        self,
        tokens: list[list[int]],
        prompt_tokens: list[list[int]],
        prompt_features: torch.Tensor,
        prompt_features_lens: torch.Tensor,
        features_lens: torch.Tensor | None = None,
        speed: float = 1.0,
        t_shift: float = 1.0,
        duration: str = "predict",
        num_step: int = 5,
        guidance_scale: float = 0.5,
    ) -> torch.Tensor:
        """Generate acoustic features from text tokens and a voice prompt.

        Args:
            tokens: Target text token ID sequences, one list per batch item.
            prompt_tokens: Prompt transcription token ID sequences.
            prompt_features: Prompt log-mel features of shape (B, T, feat_dim).
            prompt_features_lens: Prompt feature lengths, shape (B,).
            features_lens: Ground-truth target feature lengths, shape (B,).
                Used only when ``duration="real"``.
            speed: Speaking rate multiplier used when ``duration="predict"``.
            t_shift: Time-shift parameter for the ODE solver schedule.
            duration: Duration mode — ``"predict"`` estimates length from
                token-count ratio; ``"real"`` uses ``features_lens`` directly.
            num_step: Number of ODE solver steps.
            guidance_scale: Classifier-free guidance scale.

        Returns:
            A tuple ``(x1_wo_prompt, x1_wo_prompt_lens, x1_prompt,
            prompt_features_lens)`` where ``x1_wo_prompt`` contains the
            generated (non-prompt) frames of shape (B, T_gen, feat_dim) and
            ``x1_wo_prompt_lens`` holds their lengths.

        Raises:
            ValueError: If ``duration`` is not ``"real"`` or ``"predict"``, or
                if ``duration="real"`` but ``features_lens`` is ``None``.
        """
        if duration not in ["real", "predict"]:
            msg = f"duration must be 'real' or 'predict', got {duration!r}"
            raise ValueError(msg)

        if duration == "predict":
            (
                text_condition,
                padding_mask,
            ) = self.forward_text_inference_ratio_duration(
                tokens=tokens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
                speed=speed,
            )
        else:
            if features_lens is None:
                msg = "features_lens must be provided when duration='real'"
                raise ValueError(msg)
            text_condition, padding_mask = self.forward_text_inference_gt_duration(
                tokens=tokens,
                features_lens=features_lens,
                prompt_tokens=prompt_tokens,
                prompt_features_lens=prompt_features_lens,
            )
        batch_size, num_frames, _ = text_condition.shape

        speech_condition = torch.nn.functional.pad(
            prompt_features, (0, 0, 0, num_frames - prompt_features.size(1))
        )  # (B, T, F)

        # False means speech condition positions.
        speech_condition_mask = make_pad_mask(prompt_features_lens, num_frames)
        speech_condition = torch.where(
            speech_condition_mask.unsqueeze(-1),
            torch.zeros_like(speech_condition),
            speech_condition,
        )

        x0 = torch.randn(
            batch_size,
            num_frames,
            prompt_features.size(-1),
            device=text_condition.device,
        )

        x1 = self.solver.sample(
            x=x0,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_shift=t_shift,
        )
        x1_wo_prompt_lens = (~padding_mask).sum(-1) - prompt_features_lens
        x1_prompt = torch.zeros(x1.size(0), prompt_features_lens.max(), x1.size(2), device=x1.device)
        x1_wo_prompt = torch.zeros(x1.size(0), x1_wo_prompt_lens.max(), x1.size(2), device=x1.device)
        for i in range(x1.size(0)):
            x1_wo_prompt[i, : x1_wo_prompt_lens[i], :] = x1[
                i,
                prompt_features_lens[i] : prompt_features_lens[i] + x1_wo_prompt_lens[i],
            ]
            x1_prompt[i, : prompt_features_lens[i], :] = x1[i, : prompt_features_lens[i]]

        return x1_wo_prompt, x1_wo_prompt_lens, x1_prompt, prompt_features_lens

    def sample_intermediate(  # noqa: PLR0913
        self,
        tokens: list[list[int]],
        features: torch.Tensor,
        features_lens: torch.Tensor,
        noise: torch.Tensor,
        speech_condition_mask: torch.Tensor,
        t_start: float,
        t_end: float,
        num_step: int = 1,
        guidance_scale: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate acoustic features over an intermediate ODE time interval.

        Args:
            tokens: Text token ID sequences, one list per batch item.
            features: Acoustic features of shape (B, T, feat_dim).
            features_lens: Length of each feature sequence, shape (B,).
            noise: Initial noise tensor of shape (B, T, feat_dim).
            speech_condition_mask: Boolean mask of shape (B, T); ``True``
                positions are non-condition (to be generated).
            t_start: ODE integration start timestep in [0, 1].
            t_end: ODE integration end timestep in [0, 1].
            num_step: Number of solver steps within ``[t_start, t_end]``.
            guidance_scale: Classifier-free guidance scale of shape (B, 1, 1),
                or ``None`` to disable guidance.

        Returns:
            A tuple ``(x_t_end, x_t_end_lens)`` where ``x_t_end`` has shape
            (B, T, feat_dim) and ``x_t_end_lens`` is the unpadded length of
            each sequence, shape (B,).
        """
        (
            text_condition,
            padding_mask,
        ) = self.forward_text_train(
            tokens=tokens,
            features_lens=features_lens,
        )

        speech_condition = torch.where(speech_condition_mask.unsqueeze(-1), 0, features)

        x_t_end = self.solver.sample(
            x=noise,
            text_condition=text_condition,
            speech_condition=speech_condition,
            padding_mask=padding_mask,
            num_step=num_step,
            guidance_scale=guidance_scale,
            t_start=t_start,
            t_end=t_end,
        )
        x_t_end_lens = (~padding_mask).sum(-1)
        return x_t_end, x_t_end_lens


# end zipvoice/models/zipvoice.py
