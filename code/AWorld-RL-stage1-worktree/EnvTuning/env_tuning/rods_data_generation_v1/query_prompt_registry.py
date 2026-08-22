"""Fail-closed registry for reconstructed BFCL per-class query prompts."""

from __future__ import annotations

from typing import Any, Mapping

from .prompts import load_prompt


PUBLIC_BFCL_CLASSES = (
    "GorillaFileSystem",
    "MathAPI",
    "MessageAPI",
    "TwitterAPI",
    "TicketAPI",
    "TradingBot",
    "TravelAPI",
    "VehicleControlAPI",
)


class QueryPromptRegistry:
    """Render Appendix-D per-class conditioning without claiming official text.

    RODS publishes the per-class mechanism but not these class prompt bodies;
    every file selected here is therefore explicitly RECONSTRUCTED.
    """

    SOURCE_STATUS = "RECONSTRUCTED_FROM_RODS_SPEC"

    def render(self, class_name: str, values: Mapping[str, Any]) -> str:
        if class_name not in PUBLIC_BFCL_CLASSES:
            raise FileNotFoundError(
                f"no reconstructed per-class query prompt for BFCL class {class_name!r}"
            )
        guidance = load_prompt(
            f"reconstructed/query_generation/{class_name}.txt"
        )
        return load_prompt(
            "reconstructed/query_generation/base.txt",
            {**dict(values), "class_name": class_name, "class_guidance": guidance},
        )
