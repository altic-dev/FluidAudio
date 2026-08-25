"""CoreML-compatibility patches for the HF Granite Speech 5 encoder.

Each patch replaces an op that coremltools cannot lower with a numerically
identical formulation. Nothing here changes the model's arithmetic.
"""

from __future__ import annotations

import torch
from transformers.models.granite_speech5 import modeling_granite_speech5 as gs5


def _pool_by_two(hidden_states: torch.Tensor) -> torch.Tensor:
    """Mean-pool pairs of adjacent time steps, dropping a ragged last frame.

    Upstream uses ``hidden_states.unfold(1, 2, 2).mean(-1)``; coremltools has no
    lowering for ``unfold``. A reshape over non-overlapping pairs is the same
    computation for step == size.
    """
    batch, length, dim = hidden_states.shape
    half = length // 2
    trimmed = hidden_states[:, : 2 * half]
    return trimmed.reshape(batch, half, 2, dim).mean(dim=2)


def _subsampling_block_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_embeddings: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    residual = hidden_states
    hidden_states = self.feed_forward1(self.norm_feed_forward1(hidden_states))
    hidden_states = residual + 0.5 * hidden_states

    normalized_hidden_states = self.norm_self_att(hidden_states)
    attn_output, _ = self.self_attn(
        hidden_states=normalized_hidden_states,
        attention_mask=attention_mask,
        position_embeddings=position_embeddings,
        **kwargs,
    )
    hidden_states = hidden_states + attn_output

    conv_output = self.conv(self.norm_conv(hidden_states), attention_mask=attention_mask)
    pooled = _pool_by_two(hidden_states)
    hidden_states = pooled + conv_output[:, : pooled.shape[1]]

    ff2_output = self.feed_forward2(self.norm_feed_forward2(hidden_states))
    hidden_states = hidden_states + 0.5 * ff2_output

    return self.norm_out(hidden_states)


def apply() -> None:
    """Install every patch. Safe to call more than once."""
    gs5.GraniteSpeech5EncoderSubsamplingBlock.forward = _subsampling_block_forward
