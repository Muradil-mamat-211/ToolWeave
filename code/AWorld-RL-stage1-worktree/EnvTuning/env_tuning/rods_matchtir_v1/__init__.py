"""RODS global advantage plus ToolWeave runtime-interaction local credit."""

from importlib import import_module

__all__ = [
    "CanonicalToolCall",
    "LocalCreditConfig",
    "fuse_rods_and_local_advantages",
    "hard_match_calls",
    "matchtir_similarity",
]

_EXPORT_MODULES = {
    "CanonicalToolCall": "matching",
    "LocalCreditConfig": "advantage",
    "fuse_rods_and_local_advantages": "advantage",
    "hard_match_calls": "matching",
    "matchtir_similarity": "matching",
}


def __getattr__(name):
    # Generator only needs lifecycle/provenance. Loading those modules must
    # not require Training's torch/scipy dependencies or initialize a GPU.
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value
    return value
