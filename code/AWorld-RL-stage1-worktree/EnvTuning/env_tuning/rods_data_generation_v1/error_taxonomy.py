"""RODS Appendix E error taxonomy and recovery policy."""

from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    PARAM_GEN_FAILED = "param_gen_failed"
    DECOMPOSE_FAILED = "decompose_failed"
    FUNC_SAMPLE_FAILED = "func_sample_failed"
    VM_EXEC_FAILED = "vm_exec_failed"
    DUPLICATE_FUNC = "duplicate_func"
    QUERY_GEN_FAILED = "query_gen_failed"
    QUERY_VERIFY_FAILED = "query_verify_failed"
    QUERY_VERIFY_NO_TAG = "query_verify_no_tag"
    CONVERSATION_CONSTRUCT_FAILED = "conversation_construct_failed"
    NO_PROMPTS = "no_prompts"
    NO_PATTERN = "no_pattern"
    PIPELINE_EXCEPTION = "pipeline_exception"


PATCHABLE_ERRORS = frozenset(
    {
        ErrorType.PARAM_GEN_FAILED,
        ErrorType.DECOMPOSE_FAILED,
        ErrorType.FUNC_SAMPLE_FAILED,
        ErrorType.VM_EXEC_FAILED,
    }
)


ERROR_GUIDANCE: dict[ErrorType, str] = {
    ErrorType.PARAM_GEN_FAILED: "Use functions whose required parameters can be grounded in the available state.",
    ErrorType.DECOMPOSE_FAILED: "Use BOTTOM-LEVEL functions only.",
    ErrorType.FUNC_SAMPLE_FAILED: "AVOID functions requiring authentication or specific prior state.",
    ErrorType.VM_EXEC_FAILED: "Avoid the failed execution path and choose functions compatible with the patched state.",
    ErrorType.DUPLICATE_FUNC: "Do not repeat the same function call within a turn.",
    ErrorType.QUERY_GEN_FAILED: "Use simpler function combinations.",
    ErrorType.QUERY_VERIFY_FAILED: "Use 1 function per turn to simplify.",
    ErrorType.QUERY_VERIFY_NO_TAG: "Use a plan whose intent can be stated and verified unambiguously.",
    ErrorType.CONVERSATION_CONSTRUCT_FAILED: "Use independent, clearly ordered turns with resolvable dependencies.",
    ErrorType.NO_PROMPTS: "Choose a class supported by the configured prompt/catalog set.",
    ErrorType.NO_PATTERN: "Choose a class order with a valid deterministic execution pattern.",
    ErrorType.PIPELINE_EXCEPTION: "Generate a simpler, completely different plan.",
}
