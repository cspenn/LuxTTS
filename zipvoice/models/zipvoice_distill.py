"""ZipVoiceDistill: consistency-distilled TTS model for fast synthesis."""

# start zipvoice/models/zipvoice_distill.py
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

from zipvoice.models.modules.solver import DistillEulerSolver
from zipvoice.models.modules.zipformer import TTSZipformer
from zipvoice.models.zipvoice import ZipVoice


class ZipVoiceDistill(ZipVoice):
    """Flow-matching TTS model trained with consistency distillation.

    Extends ``ZipVoice`` by replacing the standard Euler solver with a
    distilled ``DistillEulerSolver``, enabling high-quality synthesis in
    very few ODE steps (typically 1–4).  The flow-matching decoder is
    rebuilt with an additional guidance-scale embedding so that
    classifier-free guidance can be applied efficiently during distillation.
    """

    def __init__(self, *args, **kwargs):
        """Initialize ZipVoiceDistill, replacing the solver with DistillEulerSolver."""
        super().__init__(*args, **kwargs)

        required_params = {
            "feat_dim",
            "fm_decoder_downsampling_factor",
            "fm_decoder_num_layers",
            "fm_decoder_cnn_module_kernel",
            "fm_decoder_dim",
            "fm_decoder_feedforward_dim",
            "fm_decoder_num_heads",
            "query_head_dim",
            "pos_head_dim",
            "value_head_dim",
            "pos_dim",
            "time_embed_dim",
        }

        missing = [p for p in required_params if p not in kwargs]
        if missing:
            msg = f"Missing required parameters: {', '.join(missing)}"
            raise ValueError(msg)

        self.fm_decoder = TTSZipformer(
            in_dim=kwargs["feat_dim"] * 3,
            out_dim=kwargs["feat_dim"],
            downsampling_factor=kwargs["fm_decoder_downsampling_factor"],
            num_encoder_layers=kwargs["fm_decoder_num_layers"],
            cnn_module_kernel=kwargs["fm_decoder_cnn_module_kernel"],
            encoder_dim=kwargs["fm_decoder_dim"],
            feedforward_dim=kwargs["fm_decoder_feedforward_dim"],
            num_heads=kwargs["fm_decoder_num_heads"],
            query_head_dim=kwargs["query_head_dim"],
            pos_head_dim=kwargs["pos_head_dim"],
            value_head_dim=kwargs["value_head_dim"],
            pos_dim=kwargs["pos_dim"],
            use_time_embed=True,
            time_embed_dim=kwargs["time_embed_dim"],
            use_guidance_scale_embed=True,
        )
        self.solver = DistillEulerSolver(self, func_name="forward_fm_decoder")

    def forward(  # noqa: PLR0913
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
        """Run the distillation forward pass for a single time-interval step.

        Delegates to ``sample_intermediate`` to integrate the ODE from
        ``t_start`` to ``t_end`` using the distilled solver.

        Args:
            tokens: Text token ID sequences, one list per batch item.
            features: Acoustic features of shape (B, T, feat_dim).
            features_lens: Length of each feature sequence, shape (B,).
            noise: Initial noise tensor of shape (B, T, feat_dim).
            speech_condition_mask: Boolean mask of shape (B, T); ``True``
                positions are non-condition (to be generated).
            t_start: Start timestep for this ODE interval (in [0, 1]).
            t_end: End timestep for this ODE interval (in [0, 1]).
            num_step: Number of solver steps within ``[t_start, t_end]``.
            guidance_scale: Classifier-free guidance scale tensor of shape
                (B, 1, 1), or ``None`` to disable guidance.

        Returns:
            Predicted acoustic features at timestep ``t_end``, shape
            (B, T, feat_dim).
        """
        return self.sample_intermediate(
            tokens=tokens,
            features=features,
            features_lens=features_lens,
            noise=noise,
            speech_condition_mask=speech_condition_mask,
            t_start=t_start,
            t_end=t_end,
            num_step=num_step,
            guidance_scale=guidance_scale,
        )


# end zipvoice/models/zipvoice_distill.py
