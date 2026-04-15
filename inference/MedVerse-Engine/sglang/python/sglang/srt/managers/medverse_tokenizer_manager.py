"""MedVerseTokenizerManager – extends Multiverse's TokenizerManager.

Phase I requests are passed through as-is (standard autoregressive decoding).
The only addition: ensure </Plan> is in the stop strings so that sglang's
sampling loop naturally halts when the LLM finishes its planning block.

The MedVerseScheduler takes over from there (runtime fork on </Plan> detection).
"""

from __future__ import annotations

import logging
from typing import Union

from sglang.srt.managers.io_struct import (
    EmbeddingReqInput,
    GenerateReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)

# Stop string that signals Phase I is complete.
# MedVerse14k data always has </Plan> immediately after the last <Outline> tag.
_MEDVERSE_STOP_STRINGS = ["</Plan>"]


class MedVerseTokenizerManager(TokenizerManager):
    """Extends Multiverse's TokenizerManager to support the MedVerse pipeline."""

    async def _tokenize_one_request(
        self,
        obj: Union[GenerateReqInput, EmbeddingReqInput],
    ):
        """Tokenize request and inject </Plan> into stop strings."""
        tokenized_obj = await super()._tokenize_one_request(obj)

        if not isinstance(tokenized_obj, TokenizedGenerateReqInput):
            return tokenized_obj

        sp = tokenized_obj.sampling_params
        if sp is None:
            return tokenized_obj

        # Inject </Plan> stop string ONLY for MedVerse requests.
        # Skip injection if:
        #   1. Request doesn't contain <Think> in the prompt (AR baseline), OR
        #   2. Request has custom_params["no_fork"] = True (explicit AR baseline flag)
        input_text = getattr(tokenized_obj, "input_text", "") or ""
        custom = getattr(sp, "custom_params", None) or {}
        is_no_fork = bool(custom.get("no_fork", False))
        is_medverse_request = ("<Think>" in input_text) and not is_no_fork

        if not is_medverse_request:
            # AR baseline or no-fork request – no injection
            return tokenized_obj

        # Inject </Plan> stop string so Phase I generation halts at the right point
        # NOTE: After normalize(), SamplingParams stores stop strings in stop_strs
        # (not stop). We must update stop_strs directly.
        existing_stops = list(sp.stop_strs) if sp.stop_strs else []
        added = False
        for stop in _MEDVERSE_STOP_STRINGS:
            if stop not in existing_stops:
                existing_stops.append(stop)
                added = True
        if added:
            sp.stop_strs = existing_stops
            # Update stop_str_max_len so the scheduler checks long enough suffixes
            plan_end_ids = self.tokenizer.encode("</Plan>", add_special_tokens=False)
            sp.stop_str_max_len = max(
                getattr(sp, "stop_str_max_len", 0),
                len(plan_end_ids),
            )
            logger.info(
                f"[MedVerse] Injected stop strings for {tokenized_obj.rid}: "
                f"{existing_stops}, stop_str_max_len={sp.stop_str_max_len}"
            )

        return tokenized_obj
