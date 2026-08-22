import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any, Union
import ast

from .data_models import AttemptedActionType

def parse_query_response_prompting(api_response: str) -> dict:
        #TODO parsing the future thinking tag in the api_response
        resp_arr = api_response.split('</think>')
        if len(resp_arr) > 1:
            response = resp_arr[-1].strip()
        else:
            # 没有think标志位则用整个输出
            # response = "response outputs too long or no </think> in response."
            response = api_response
            
            print("resp_arr", resp_arr)
        return {
            "model_responses": response
        }


@dataclass(frozen=True)
class ModelResponseParseResult:
    """Strict protocol parse result with independent action/call semantics."""

    content: str
    message: str
    attempted_action_type: AttemptedActionType
    action_classification_reliable: bool
    call_parse_reliable: bool
    error_code: Optional[str] = None


def parse_model_response_detailed(response: str) -> ModelResponseParseResult:
    """Parse one assistant action and retain parser-grounded failure semantics.

    Action intent is classified from the same complete-tag matches used by the
    executable protocol parser.  No raw substring heuristic is used.  A
    protocol- or JSON-rejected tool block therefore remains a reliable tool
    attempt while its structured call parse is marked unsuccessful.
    """

    raw = response
    response = response.strip()
    thinking_matches = re.findall(
        r"<think>([\s\S]*?)</think>", response, flags=re.DOTALL
    )
    tool_matches = re.findall(
        r"<tool_call>([\s\S]*?)</tool_call>", response, flags=re.DOTALL
    )
    answer_matches = re.findall(
        r"<answer>([\s\S]*?)</answer>", response, flags=re.DOTALL
    )

    has_tool_call = len(tool_matches) > 0
    has_answer = len(answer_matches) > 0
    if has_tool_call and not has_answer:
        attempted_action_type = AttemptedActionType.TOOL_CALL
        action_classification_reliable = True
    elif has_answer and not has_tool_call:
        attempted_action_type = AttemptedActionType.ANSWER
        action_classification_reliable = True
    else:
        attempted_action_type = AttemptedActionType.UNKNOWN
        action_classification_reliable = False

    def error(message: str, code: str) -> ModelResponseParseResult:
        return ModelResponseParseResult(
            content=raw,
            message=message,
            attempted_action_type=attempted_action_type,
            action_classification_reliable=action_classification_reliable,
            call_parse_reliable=False,
            error_code=code,
        )

    # Preserve the legacy parser's validation order and exact messages.
    if len(thinking_matches) == 0:
        return error("Error: Missing <think></think> tags", "missing_think")
    if len(thinking_matches) > 1:
        return error(
            "Error: Multiple <think></think> tag pairs found. Only one pair is allowed.",
            "multiple_think_blocks",
        )
    if len(tool_matches) > 1:
        return error(
            "Error: Multiple <tool_call></tool_call> tag pairs found. Only one pair is allowed.",
            "multiple_tool_call_blocks",
        )
    if len(answer_matches) > 1:
        return error(
            "Error: Multiple <answer></answer> tag pairs found. Only one pair is allowed.",
            "multiple_answer_blocks",
        )
    if has_tool_call and has_answer:
        return error(
            "Error: Response cannot contain both <tool_call> and <answer> tags",
            "mixed_action_blocks",
        )
    if not has_tool_call and not has_answer:
        return error(
            "Error: Response must contain either <tool_call> or <answer> tags",
            "missing_action_block",
        )

    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.DOTALL)
    if has_tool_call:
        cleaned = re.sub(
            r"<tool_call>[\s\S]*?</tool_call>", "", cleaned, flags=re.DOTALL
        )
    else:
        cleaned = re.sub(r"<answer>[\s\S]*?</answer>", "", cleaned, flags=re.DOTALL)
    if cleaned.strip():
        return error(
            "Error: Response must not contain text outside the required XML tags",
            "outside_required_tags",
        )

    if has_tool_call:
        tool_body = tool_matches[0].strip()
        try:
            obj = json.loads(tool_body)
        except json.JSONDecodeError as exc:
            return error(
                f"Error: Invalid JSON inside <tool_call>: {exc}",
                "invalid_tool_json",
            )
        return ModelResponseParseResult(
            content=json.dumps(obj),
            message="tool_call",
            attempted_action_type=AttemptedActionType.TOOL_CALL,
            action_classification_reliable=True,
            call_parse_reliable=True,
        )

    return ModelResponseParseResult(
        content=answer_matches[0].strip(),
        message="answer",
        attempted_action_type=AttemptedActionType.ANSWER,
        action_classification_reliable=True,
        call_parse_reliable=False,
    )


def parse_model_response(
    response: str,
) -> Tuple[str, str]:
    """
    Parse LLM response that must follow one of the two formats
    (thinking+tool_call) or (thinking+answer).

    Returns
    -------
    content : Union[str, list]
        - If <tool_call> is present: the parsed JSON (list) inside the tag.
        - If <answer>    is present: the string inside the tag (stripped).
        - If format error: the original response.
    msg : str
        "answer" or "tool_all"  if success;
        error description (English) on failure.
    """

    parsed = parse_model_response_detailed(response)
    return parsed.content, parsed.message



def is_empty_execute_response(input_list: list):
    if len(input_list) == 0:
        return True
    if len(input_list) == 1 and len(input_list[0]) == 0:
        return True
    return False


def resolve_ast_call(elem):
    # Handle nested attributes for deeply nested module paths
    func_parts = []
    func_part = elem.func
    while isinstance(func_part, ast.Attribute):
        func_parts.append(func_part.attr)
        func_part = func_part.value
    if isinstance(func_part, ast.Name):
        func_parts.append(func_part.id)
    func_name = ".".join(reversed(func_parts))
    args_dict = {}
    for arg in elem.keywords:
        output = resolve_ast_by_type(arg.value)
        args_dict[arg.arg] = output
    return {func_name: args_dict}


def resolve_ast_by_type(value):
    if isinstance(value, ast.Constant):
        if value.value is Ellipsis:
            output = "..."
        else:
            output = value.value
    elif isinstance(value, ast.UnaryOp):
        output = -value.operand.value
    elif isinstance(value, ast.List):
        output = [resolve_ast_by_type(v) for v in value.elts]
    elif isinstance(value, ast.Dict):
        output = {
            resolve_ast_by_type(k): resolve_ast_by_type(v)
            for k, v in zip(value.keys, value.values)
        }
    elif isinstance(
        value, ast.NameConstant
    ):  # Added this condition to handle boolean values
        output = value.value
    elif isinstance(
        value, ast.BinOp
    ):  # Added this condition to handle function calls as arguments
        output = eval(ast.unparse(value))
    elif isinstance(value, ast.Name):
        output = value.id
    elif isinstance(value, ast.Call):
        if len(value.keywords) == 0:
            output = ast.unparse(value)
        else:
            output = resolve_ast_call(value)
    elif isinstance(value, ast.Tuple):
        output = tuple(resolve_ast_by_type(v) for v in value.elts)
    elif isinstance(value, ast.Lambda):
        output = eval(ast.unparse(value.body[0].value))
    elif isinstance(value, ast.Ellipsis):
        output = "..."
    elif isinstance(value, ast.Subscript):
        try:
            output = ast.unparse(value.body[0].value)
        except:
            output = ast.unparse(value.value) + "[" + ast.unparse(value.slice) + "]"
    else:
        raise Exception(f"Unsupported AST type: {type(value)}")
    return output

def ast_parse(input_str, language="Python"):
    if language == "Python":
        cleaned_input = input_str.strip("[]'")
        parsed = ast.parse(cleaned_input, mode="eval")
        extracted = []
        if isinstance(parsed.body, ast.Call):
            extracted.append(resolve_ast_call(parsed.body))
        else:
            for elem in parsed.body.elts:
                assert isinstance(elem, ast.Call)
                extracted.append(resolve_ast_call(elem))
        return extracted
    else:
        raise NotImplementedError(f"Unsupported language: {language}. Only support Python language by default.")


def parse_nested_value(value):
    """
    Parse a potentially nested value from the AST output.

    Args:
        value: The value to parse, which could be a nested dictionary, which includes another function call, or a simple value.

    Returns:
        str: A string representation of the value, handling nested function calls and nested dictionary function arguments.
    """
    if isinstance(value, dict):
        # Check if the dictionary represents a function call (i.e., the value is another dictionary or complex structure)
        if all(isinstance(v, dict) for v in value.values()):
            func_name = list(value.keys())[0]
            args = value[func_name]
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in args.items())
            return f"{func_name}({args_str})"
        else:
            # If it's a simple dictionary, treat it as key-value pairs
            return (
                "{"
                + ", ".join(f"'{k}': {parse_nested_value(v)}" for k, v in value.items())
                + "}"
            )
    return repr(value)


def decoded_output_to_execution_list(decoded_output):
    """
    Convert decoded output to a list of executable function calls.

    Args:
        decoded_output (list): A list of dictionaries representing function calls.

    Returns:
        list: A list of strings, each representing an executable function call.
    """
    execution_list = []
    for function_call in decoded_output:
        for key, value in function_call.items():
            args_str = ", ".join(f"{k}={parse_nested_value(v)}" for k, v in value.items())
            execution_list.append(f"{key}({args_str})")
    return execution_list



def default_decode_execute_prompting(result):
    result = result.strip("`\n ")
    if not result.startswith("["):
        result = "[" + result
    if not result.endswith("]"):
        result = result + "]"
    decoded_output = ast_parse(result)
    return decoded_output_to_execution_list(decoded_output)


def _build_call_str(name: str, args: Dict[str, Any]) -> str:
    """将函数名和参数字典转成可读形式 'func(a=1, b=\"x\")'。"""
    args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
    return f"{name}({args_str})"


def parse_tool_call_objects(raw: str) -> List[Dict[str, Any]]:
    """Return structured individual calls from the already-validated JSON body.

    This is the single parser boundary shared by execution and MatchTIR
    provenance.  Invalid array elements are preserved as ``valid=False`` for
    local-credit diagnostics, while :func:`parse_tool_calls` continues to omit
    them from environment execution exactly as before.
    """

    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    calls = decoded if isinstance(decoded, list) else [decoded]
    structured: List[Dict[str, Any]] = []
    for call_idx, call in enumerate(calls):
        if not isinstance(call, dict):
            structured.append(
                {
                    "call_idx": call_idx,
                    "name": "",
                    "arguments": {},
                    "valid": False,
                }
            )
            continue
        name = call.get("name")
        valid_name = isinstance(name, str) and bool(name.strip())
        arguments = call.get("arguments")
        # Preserve the execution parser's established normalization: absent,
        # null, or non-object arguments become an empty object.
        if not isinstance(arguments, dict):
            arguments = {}
        structured.append(
            {
                "call_idx": call_idx,
                "name": name.strip() if valid_name else "",
                "arguments": arguments,
                "valid": valid_name,
            }
        )
    return structured


def parse_tool_calls(raw: str
                     ) -> str:
    """
    """
    call_strings: List[str] = []
    for call in parse_tool_call_objects(raw):
        if not call["valid"]:
            continue
        call_strings.append(_build_call_str(call["name"], call["arguments"]))

    # call_strings = [_build_call_str(c["name"], c["arguments"]) for c in calls]
    # formatted_calls = "[" + ", ".join(call_strings) + "]"
    
    return "[" + ", ".join(call_strings) + "]" if call_strings else "[]"

def has_execution_error(execution_results: list[str]) -> bool:
    """
    Return True if any result in `execution_results` indicates a failure.

    A failure string is produced by `execute_multi_turn_func_call` when an
    exception occurs, and always starts with the prefix:
        "Error during execution: "

    Parameters
    ----------
    execution_results : list[str]
        List returned by `execute_multi_turn_func_call`.

    Returns
    -------
    bool
        True  – at least one function call failed  
        False – all function calls succeeded
    """
    error_prefix = "Error during execution:"
    return any(
        isinstance(res, str) and res.startswith(error_prefix)
        for res in execution_results
    )


def check_execution_results(execution_results: List[Any]) -> Tuple[bool, List[Any]]:
    """
    检测 execution_results 中的失败项。

    Parameters
    ----------
    execution_results : List[Any]
        execute_multi_turn_func_call 返回的 execution_results。

    Returns
    -------
    has_error : bool
        只要存在一项失败则为 True，否则为 False。
    failed_items : List[Any]
        所有被判定为失败的条目（原样返回，便于后续排查）。
    """
    error_prefix = "Error during execution:"

    def is_failure(item: Any) -> bool:
        #easy tool call
        if isinstance(item, str) and item.startswith(error_prefix):
            return True

        #hard tool call
        if isinstance(item, str) and item.lstrip().startswith("{"):
            # 3a) 先尝试用 json 解析
            try:
                obj = json.loads(item)
                if isinstance(obj, dict) and "error" in obj:
                    return True
            except json.JSONDecodeError:
                if "'error':" in item or '"error":' in item:
                    return True

        return False

    failed_items = [item for item in execution_results if is_failure(item)]
    has_error = bool(failed_items)
    return has_error, failed_items


if __name__ == "__main__":
    st = """[authenticate_travel(client_id=\"discover3rID9537\", client_secret=\"K3yToSecrecy!\", refresh_token=\"updat3Mofresh\", grant_type=\"write\", user_first_name=\"James\", user_last_name=\"Montgomery\"), book_flight(access_token=\"886764\", card_id=\"card_8911\", travel_date=\"2024-01-01\", travel_from=\"RMS\", travel_to=\"OKD\", travel_class=\"first\", travel_cost=2700.0)]""" 
    print(default_decode_execute_prompting(st))
    raw_calls = '[{"name": "authenticate_travel", "arguments": {"client_id": "discover3rID9537", "client_secret": "K3yToSecrecy!"}}, {"name": "run_up_down"}]'
    formatted_tool_calls = parse_tool_calls(raw_calls)
    
    print(formatted_tool_calls)
