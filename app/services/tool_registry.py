import json
import inspect
from typing import Any, Dict, List, Optional, Callable
import logging

from pydantic import TypeAdapter

from app.services.math_service import (
    calculate, solve_equation, differentiate, integrate,
    definite_integral, compute_limit, partial_fractions,
    simplify_expression, taylor_series, solve_system,
    mean, variance, standard_deviation, median, mode,
    correlation_coefficient, linear_regression, matrix_addition,
    matrix_multiplication, matrix_transpose, matrix_determinant,
    matrix_eigenvalues, vector_dot_product, vector_cross_product,
    vector_magnitude, summation, expand, factorize
)

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
    "matrix_eigenvalues": matrix_eigenvalues,
    "vector_dot_product": vector_dot_product,
    "vector_cross_product": vector_cross_product,
    "vector_magnitude": vector_magnitude,
    "summation": summation,
    "expand": expand,
    "factorize": factorize,
}

def generate_tool_schema(func: Callable) -> Dict[str, Any]:
    """Auto-generates OpenAI tool schema from a function signature using Pydantic."""
    sig = inspect.signature(func)
    doc = func.__doc__ or ""
    
    # Extract the short description (first non-empty line)
    description = ""
    for line in doc.split('\n'):
        line = line.strip()
        if line and not line.startswith(('Args:', 'Returns:', 'Examples:', 'Notes:', '>>>')):
            description = line
            break
            
    properties = {}
    required = []
    
    for name, param in sig.parameters.items():
        # Handle lack of annotations gracefully by falling back to str
        annotation = param.annotation if param.annotation != inspect.Parameter.empty else str
        
        # Use Pydantic's TypeAdapter to generate the JSON schema for this specific type
        adapter = TypeAdapter(annotation)
        param_schema = adapter.json_schema()
        
        # Patch for OpenAI function calling array schema requiring 'items'
        def patch_schema(schema: dict) -> dict:
            if isinstance(schema, dict):
                if schema.get("type") == "array":
                    if "prefixItems" in schema:
                        # Provide a fallback items definition for tuples and remove prefixItems
                        schema["items"] = schema["prefixItems"][0] if schema["prefixItems"] else {}
                        del schema["prefixItems"]
                    elif "items" not in schema:
                        schema["items"] = {}
                # Avoid dictionary changed size during iteration
                for v in list(schema.values()):
                    if isinstance(v, dict):
                        patch_schema(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, dict):
                                patch_schema(item)
            return schema

        param_schema = patch_schema(param_schema)
        
        if param.default == inspect.Parameter.empty:
            required.append(name)
        else:
            param_schema["default"] = param.default
            
        properties[name] = param_schema
        
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description or f"Execute {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False
            },
        },
    }

def get_tools_for_openai() -> List[Dict[str, Any]]:
    """Dynamically build OpenAI tool schemas from all registered executors."""
    return [generate_tool_schema(func) for func in _EXECUTORS.values()]

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
