import json
from typing import Any, Dict, List, Optional
import logging
 
from calculator_server import (
    calculate,
    solve_equation,
    differentiate,
    integrate,
    mean,
    variance,
    standard_deviation,
    median,
    mode,
    correlation_coefficient,
    linear_regression,
    matrix_addition,
    matrix_multiplication,
    matrix_transpose,
    matrix_determinant,
    vector_dot_product,
    vector_cross_product,
    vector_magnitude,
    summation,
    expand,
    factorize,
)


def get_tools_for_openai() -> List[Dict[str, Any]]:
    # Map of MCP tool -> OpenAI tool schema (function calling)
    return [
        {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Evaluate a mathematical expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "solve_equation",
                "description": "Solve an equation for x.",
                "parameters": {
                    "type": "object",
                    "properties": {"equation": {"type": "string"}},
                    "required": ["equation"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "differentiate",
                "description": "Differentiate an expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "variable": {"type": "string", "default": "x"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "integrate",
                "description": "Integrate an expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "variable": {"type": "string", "default": "x"},
                    },
                    "required": ["expression"],
                },
            },
        },
        # Stats and matrix/vector tools shortened for brevity but included
        {
            "type": "function",
            "function": {
                "name": "mean",
                "description": "Compute mean of a list of numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "array", "items": {"type": "number"}}},
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "variance",
                "description": "Compute variance of a list of numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "array", "items": {"type": "number"}}},
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "standard_deviation",
                "description": "Compute standard deviation of a list of numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "array", "items": {"type": "number"}}},
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "median",
                "description": "Compute median.",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "array", "items": {"type": "number"}}},
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "mode",
                "description": "Compute mode.",
                "parameters": {
                    "type": "object",
                    "properties": {"data": {"type": "array", "items": {"type": "number"}}},
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "correlation_coefficient",
                "description": "Pearson correlation coefficient of two lists.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data_x": {"type": "array", "items": {"type": "number"}},
                        "data_y": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["data_x", "data_y"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "linear_regression",
                "description": "Linear regression slope and intercept.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        }
                    },
                    "required": ["data"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "matrix_addition",
                "description": "Add two matrices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matrix_a": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "matrix_b": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                    "required": ["matrix_a", "matrix_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "matrix_multiplication",
                "description": "Multiply two matrices.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matrix_a": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                        "matrix_b": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                    "required": ["matrix_a", "matrix_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "matrix_transpose",
                "description": "Transpose a matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                    "required": ["matrix"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "matrix_determinant",
                "description": "Determinant of a matrix.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "matrix": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
                    },
                    "required": ["matrix"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vector_dot_product",
                "description": "Dot product of two vectors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vector_a": {"type": "array", "items": {"type": "number"}},
                        "vector_b": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["vector_a", "vector_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vector_cross_product",
                "description": "Cross product of two 3D vectors.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vector_a": {"type": "array", "items": {"type": "number"}},
                        "vector_b": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["vector_a", "vector_b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "vector_magnitude",
                "description": "Magnitude (L2 norm) of a vector.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "vector": {"type": "array", "items": {"type": "number"}},
                    },
                    "required": ["vector"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summation",
                "description": "Finite summation of an expression from start to end.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "start": {"type": "integer", "default": 0},
                        "end": {"type": "integer", "default": 10},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand",
                "description": "Algebraic expansion of an expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "factorize",
                "description": "Factorization of an expression.",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
    ]


_EXECUTORS = {
    "calculate": calculate,
    "solve_equation": solve_equation,
    "differentiate": differentiate,
    "integrate": integrate,
    "mean": mean,
    "variance": variance,
    "standard_deviation": standard_deviation,
    "median": median,
    "mode": mode,
    "correlation_coefficient": correlation_coefficient,
    "linear_regression": linear_regression,
    "matrix_addition": matrix_addition,
    "matrix_multiplication": matrix_multiplication,
    "matrix_transpose": matrix_transpose,
    "matrix_determinant": matrix_determinant,
    "vector_dot_product": vector_dot_product,
    "vector_cross_product": vector_cross_product,
    "vector_magnitude": vector_magnitude,
    "summation": summation,
    "expand": expand,
    "factorize": factorize,
}


async def execute_tool_call(name: Optional[str], args_json: str) -> str:
    if not name:
        return json.dumps({"error": "Missing tool name"})
    func = _EXECUTORS.get(name)
    if not func:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        parsed = json.loads(args_json or "{}")
    except Exception as exc:
        return json.dumps({"error": f"Invalid arguments JSON: {exc}"})
    try:
        logging.info("tool_call %s %s", name, parsed)
        result = func(**parsed)
        return json.dumps(result)
    except Exception as exc:
        return json.dumps({"error": str(exc)})



