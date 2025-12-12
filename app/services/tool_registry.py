import json
from typing import Any, Dict, List, Optional
import logging
 
from calculator_server import (
    calculate,
    solve_equation,
    differentiate,
    integrate,
    definite_integral,
    compute_limit,
    partial_fractions,
    simplify_expression,
    taylor_series,
    solve_system,
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
                "description": "Integrate an expression (indefinite integral).",
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
                "name": "definite_integral",
                "description": "Compute definite integral of an expression over an interval. Supports infinite bounds using 'inf' or '-inf'.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The expression to integrate"},
                        "variable": {"type": "string", "default": "x"},
                        "lower_bound": {"type": "string", "description": "Lower bound (number or 'inf'/'-inf')"},
                        "upper_bound": {"type": "string", "description": "Upper bound (number or 'inf'/'-inf')"},
                    },
                    "required": ["expression", "lower_bound", "upper_bound"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_limit",
                "description": "Compute the limit of an expression as variable approaches a point. Use '+' or '-' for one-sided limits.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The expression to take the limit of"},
                        "variable": {"type": "string", "default": "x"},
                        "point": {"type": "string", "description": "The point to approach (number or 'inf'/'-inf')"},
                        "direction": {"type": "string", "description": "Direction: '+' for right, '-' for left, '' for two-sided", "default": ""},
                    },
                    "required": ["expression", "point"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "partial_fractions",
                "description": "Perform partial fraction decomposition on a rational expression. Decomposes into sum of simpler fractions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The rational expression to decompose, e.g., '1/(x^2-1)'"},
                        "variable": {"type": "string", "default": "x"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "simplify_expression",
                "description": "Simplify a mathematical expression to its simplest form. Combines like terms, reduces fractions, simplifies trig.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The expression to simplify, e.g., '(x^2-1)/(x-1)' or 'sin(x)^2 + cos(x)^2'"},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "taylor_series",
                "description": "Compute Taylor series expansion of an expression around a point. Default is Maclaurin series (around 0).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string", "description": "The expression to expand, e.g., 'sin(x)', 'exp(x)'"},
                        "variable": {"type": "string", "default": "x"},
                        "point": {"type": "string", "description": "The point around which to expand (default: 0)", "default": "0"},
                        "order": {"type": "integer", "description": "Number of terms in the expansion", "default": 6},
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "solve_system",
                "description": "Solve a system of equations. Provide list of equations with '=' sign.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "equations": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of equations, e.g., ['x + y = 5', 'x - y = 1']"
                        },
                        "variables": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of variables to solve for",
                        },
                    },
                    "required": ["equations"],
                },
            },
        },
        # Stats and matrix/vector tools
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
    "definite_integral": definite_integral,
    "compute_limit": compute_limit,
    "partial_fractions": partial_fractions,
    "simplify_expression": simplify_expression,
    "taylor_series": taylor_series,
    "solve_system": solve_system,
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
        logging.info("🧮 EXECUTING CALCULATION: %s with args: %s", name, parsed)
        result = func(**parsed)
        result_json = json.dumps(result)
        logging.info("🧮 CALCULATION RESULT from %s: %s", name, result)
        logging.info("🧮 This result comes from SymPy/NumPy/SciPy, NOT from LLM inference!")
        return result_json
    except Exception as exc:
        error_result = {"error": str(exc)}
        logging.error("🧮 CALCULATION ERROR in %s: %s", name, error_result)
        return json.dumps(error_result)



