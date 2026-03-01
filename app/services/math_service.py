import argparse
import math
import numpy as np
from scipy import stats
from sympy import (
    symbols, solve, sympify, diff, integrate, oo, Sum, 
    limit as sympy_limit, S, apart, simplify as sympy_simplify,
    series, Eq
)
from typing import List, Tuple
import matplotlib.pyplot as plt
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from sympy import integrate as sympy_integrate
import re

# Create MCP Server

TRANSPORT = "sse"

ALLOW_FUNCTION = {
    "math": math,
    "np": np,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "sqrt": math.sqrt,
    "pi": math.pi,
    "e": math.e,
    "abs": abs,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "cot": lambda x: 1 / math.tan(x),
    "csc": lambda x: 1 / math.sin(x),
    "sec": lambda x: 1 / math.cos(x),
    "ceil": math.ceil,
    "floor": math.floor,
    "round": round,
    "factorial": math.factorial,
    "gamma": math.gamma,
    "erf": math.erf,
    "erfc": math.erfc,
    "lgamma": math.lgamma,
    "degrees": math.degrees,
    "radians": math.radians,
    "isfinite": math.isfinite,
    "isinf": math.isinf,
    "isnan": math.isnan,
    "isqrt": math.isqrt,
    "prod": np.prod,
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
    "var": np.var,
    "min": np.min,
    "max": np.max,
    "sum": np.sum,
    "cumsum": np.cumsum,
    "cumprod": np.cumprod,
    "clip": np.clip,
    "unique": np.unique,
    "sort": np.sort,
    "argsort": np.argsort,
    "argmax": np.argmax,
}


from functools import lru_cache

@lru_cache(maxsize=256)
def normalize_expression(expression: str, variable: str = "x") -> str:
    """
    Normalize common user math input into SymPy-friendly syntax.

    - Remove all whitespace to normalize string format for better cache hit rates
    - Replace caret power (^) with Python power (**)
    - Insert explicit multiplication where commonly omitted:
      3x -> 3*x, 2(x+1) -> 2*(x+1), (x+1)(x+2) -> (x+1)*(x+2), x(x+1) -> x*(x+1)

    Only handles implicit multiplication involving the primary variable and parentheses
    to avoid interfering with function names like sin(x).
    """
    s = expression or ""
    # Strip all whitespace to ensure "x + 1" and "x+1" map to same cache key
    s = re.sub(r"\s+", "", s)
    s = s.replace("^", "**")
    var = re.escape(variable)
    # Insert * between:
    # - digit and variable: 3x -> 3*x
    s = re.sub(rf"(?<=\d)\s*(?={var}\b)", "*", s)
    # - ) and variable: )x -> )*x
    s = re.sub(rf"(?<=\))\s*(?={var}\b)", "*", s)
    # - variable and (: x( -> x*(
    s = re.sub(rf"(?<={var})\s*(?=\()", "*", s)
    # - digit and (: 2( -> 2*(
    s = re.sub(r"(?<=\d)\s*(?=\()", "*", s)
    # - ) and (: )( -> )*(
    s = re.sub(r"(?<=\))\s*(?=\()", "*", s)
    return s


@lru_cache(maxsize=256)
def parse_expression(expression: str, variable: str = "x"):
    """Cached parsing of expressions to speed up SymPy operations"""
    return sympify(normalize_expression(expression, variable))

def calculate(expression: str) -> dict:
    """
    Evaluates a mathematical expression and returns the result.

    Supports basic operators (+, -, *, /, **, %), mathematical functions
    (sin, cos, tan, exp, log, log10, sqrt), and constants (pi, e).
    Uses a restricted evaluation context for safe execution.

    Args:
        expression: The mathematical expression to evaluate as a string.
                   Examples: "2 + 2", "sin(pi/4)", "sqrt(16) * 2", "log(100, 10)"

    Returns:
        On success: {"result": <calculated value>}
        On error: {"error": <error message>}

    Examples:
        >>> calculate("2 * 3 + 4")
        {'result': 10}
        >>> calculate("sin(pi/2)")
        {'result': 1.0}
        >>> calculate("sqrt(16)")
        {'result': 4.0}
        >>> calculate("invalid * expression")
        {'error': "name 'invalid' is not defined"}

    Notes:
        - Use 'x' as the variable (e.g., x**2, not x²)
        - Multiplication must be explicitly indicated with * (e.g., 2*x, not 2x)
        - Powers are represented with ** (e.g., x**2, not x^2)
    """
    try:
        expr_norm = normalize_expression(expression)
        # Safe evaluation of the expression
        result = eval(
            expr_norm,
            {"__builtins__": {}},
            ALLOW_FUNCTION,
        )
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def solve_equation(equation: str) -> dict:
    """
    Solves an algebraic equation for x and returns all solutions with verification.

    The equation must contain exactly one equality sign (=) and use a
    variable x. Can solve polynomial, trigonometric, and other equations
    supported by SymPy.

    Args:
        equation: The equation to solve as a string.
                 Format: '<left side> = <right side>'
                 Examples: "x**2 - 5*x + 6 = 0", "sin(x) = 0.5", "2*x + 3 = 7"

    Returns:
        On success: {
            "solutions": <list of solutions>,
            "equation": <original equation>,
            "num_solutions": <count>,
            "method": <solving method used>,
            "verification": [{"solution": ..., "check": ..., "valid": true/false}],
            "factored_form": <factored representation if applicable>
        }
        On error: {"error": <error message>}

    Examples:
        >>> solve_equation("x**2 - 5*x + 6 = 0")
        {'solutions': [2, 3], 'num_solutions': 2, 'verification': [...], ...}
        >>> solve_equation("2*x + 3 = 7")
        {'solutions': [2], 'num_solutions': 1, ...}
    """
    try:
        x = symbols("x")
        # Split the equation into left and right sides
        parts = equation.split("=")
        if len(parts) != 2:
            return {"error": "Equation must contain an '=' sign"}

        left = parse_expression(parts[0].strip())
        right = parse_expression(parts[1].strip())

        # Solve the equation
        solutions = solve(left - right, x)
        
        # Convert solutions to list format
        solution_list = list(solutions) if solutions else []
        
        # Determine the solving method
        expr = left - right
        if expr.is_polynomial():
            poly_degree = sp.Poly(expr, x).degree()
            if poly_degree == 1:
                method = "Linear equation"
            elif poly_degree == 2:
                method = "Quadratic equation"
            elif poly_degree == 3:
                method = "Cubic equation"
            else:
                method = f"Polynomial equation (degree {poly_degree})"
        else:
            method = "Transcendental equation"
        
        # Try to get factored form
        try:
            factored = sp.factor(expr)
            factored_form = str(factored)
        except:
            factored_form = None
        
        # Verify each solution by substituting back
        verification = []
        for sol in solution_list:
            try:
                left_result = left.subs(x, sol)
                right_result = right.subs(x, sol)
                # Check if they're equal (within floating point tolerance for numerical solutions)
                is_valid = simplify(left_result - right_result) == 0
                verification.append({
                    "solution": str(sol),
                    "left_side": str(left_result),
                    "right_side": str(right_result),
                    "valid": is_valid
                })
            except:
                verification.append({
                    "solution": str(sol),
                    "valid": None
                })
        
        return {
            "solutions": [str(s) for s in solution_list],
            "equation": equation,
            "num_solutions": len(solution_list),
            "method": method,
            "verification": verification,
            "factored_form": factored_form
        }
    except Exception as e:
        return {"error": str(e)}


def differentiate(expression: str, variable: str = "x") -> dict:
    """
    Computes the derivative of a mathematical expression with respect to a variable.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy.

    Args:
        expression: The mathematical expression to differentiate as a string.
                   Examples: "x**2", "sin(x)", "exp(x)", "log(x)"
        variable: The variable with respect to which to differentiate. Default is "x".
                 Optionally, other variables can be specified.

    Returns:
        On success: {
            "result": <derivative as string>,
            "expression": <original expression>,
            "variable": <differentiation variable>,
            "method": <differentiation rule used>,
            "interpretation": <description of what derivative represents>,
            "order": <order of derivative (first, second, etc.)>
        }
        On error: {"error": <error message>}

    Examples:
        >>> differentiate("x**2")
        {'result': '2*x', 'method': 'Power rule', ...}
        >>> differentiate("sin(x)")
        {'result': 'cos(x)', 'method': 'Trigonometric derivative', ...}
        >>> differentiate("x*y", "y")
        {'result': 'x', 'method': 'Power rule', ...}
        >>> differentiate("exp(x)")
        {'result': 'exp(x)', 'method': 'Exponential derivative', ...}

    Notes:
        - Use mathematical notation with explicit operators (* for multiplication)
        - Powers are represented with ** (e.g., x**2, not x^2)
        - For trigonometric functions, use sin(x), cos(x), etc.
        - Only support for one variable at a time (implicit differentiation not supported)
        - The derivative represents the instantaneous rate of change at any point
    """
    try:
        var = symbols(variable)
        expr = parse_expression(expression, variable)
        result = diff(expr, var)
        
        # Determine the differentiation method based on expression type
        expr_str = str(expr)
        
        if expr.is_polynomial():
            method = "Power rule: d/dx(x^n) = n*x^(n-1)"
        elif any(trig in expr_str for trig in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']):
            if 'sin' in expr_str:
                method = "Trigonometric: d/dx(sin(x)) = cos(x)"
            elif 'cos' in expr_str:
                method = "Trigonometric: d/dx(cos(x)) = -sin(x)"
            elif 'tan' in expr_str:
                method = "Trigonometric: d/dx(tan(x)) = sec²(x)"
            else:
                method = "Trigonometric derivative"
        elif 'exp' in expr_str or 'E' in expr_str:
            method = "Exponential: d/dx(e^x) = e^x"
        elif 'log' in expr_str or 'ln' in expr_str:
            method = "Logarithmic: d/dx(log(x)) = 1/x"
        elif '*' in expr_str or '**' in expr_str:
            method = "Product rule or Power rule"
        else:
            method = "Differentiation (SymPy)"
        
        # Interpretation
        interpretation = "Represents the instantaneous rate of change (slope) at any point. Used in physics for velocity/acceleration and in optimization."
        
        return {
            "result": str(result),
            "expression": expression,
            "variable": variable,
            "method": method,
            "interpretation": interpretation,
            "order": "First derivative"
        }
    except Exception as e:
        return {"error": str(e)}


def integrate(expression: str, variable: str = "x") -> dict:
    """
    Computes the indefinite integral of a mathematical expression with respect to a variable.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy.

    Args:
        expression: The mathematical expression to integrate as a string.
                   Examples: "x**2", "sin(x)", "exp(x)", "1/x"
        variable: The variable with respect to which to integrate. Default is "x".
                 Optionally, other variables can be specified.

    Returns:
        On success: {
            "result": <integral as string>,
            "expression": <original expression>,
            "variable": <integration variable>,
            "full_answer": <integral + C>,
            "method": <integration method used>,
            "note": "Add constant C for indefinite integrals",
            "interpretation": <description of what integral represents>
        }
        On error: {"error": <error message>}

    Examples:
        >>> integrate("x**2")
        {'result': 'x**3/3', 'full_answer': 'x**3/3 + C', ...}
        >>> integrate("sin(x)")
        {'result': '-cos(x)', 'full_answer': '-cos(x) + C', ...}
        >>> integrate("exp(x)")
        {'result': 'exp(x)', 'full_answer': 'exp(x) + C', ...}
        >>> integrate("1/x")
        {'result': 'log(x)', 'full_answer': 'log(x) + C', ...}

    Notes:
        - The result is the indefinite integral without the constant of integration
        - The constant C should be added for the complete answer
        - Complex expressions may be returned in simplified form
    """
    try:
        var = symbols(variable)
        expr = parse_expression(expression, variable)
        result = sympy_integrate(expr, var)
        
        # Determine the integration method based on expression type
        if expr.is_polynomial():
            method = "Power rule: integral of x^n = x^(n+1)/(n+1)"
        elif any(trig in str(expr) for trig in ['sin', 'cos', 'tan']):
            method = "Trigonometric integration"
        elif 'exp' in str(expr) or 'E' in str(expr):
            method = "Exponential integration"
        elif '1/x' in str(expr) or 'log' in str(expr):
            method = "Logarithmic integration"
        else:
            method = "Integration techniques (SymPy)"
        
        # Build the full answer with constant of integration
        full_answer = f"{str(result)} + C"
        
        # Provide interpretation
        interpretation = "Represents the antiderivative. Add constant C for the complete family of solutions."
        
        return {
            "result": str(result),
            "expression": expression,
            "variable": variable,
            "full_answer": full_answer,
            "method": method,
            "note": "Add constant C for indefinite integrals",
            "interpretation": interpretation
        }
    except Exception as e:
        return {"error": str(e)}


def definite_integral(expression: str, variable: str = "x", lower_bound: str = "0", upper_bound: str = "1") -> dict:
    """
    Computes the definite integral of a mathematical expression over an interval.

    Supports polynomials, trigonometric functions, exponential functions,
    logarithms, and other functions supported by SymPy. Supports infinite bounds
    using 'inf' or '-inf'.

    Args:
        expression: The mathematical expression to integrate as a string.
                   Examples: "x**2", "sin(x)", "exp(-x**2)"
        variable: The variable with respect to which to integrate. Default is "x".
        lower_bound: The lower bound of integration. Can be a number or 'inf'/'-inf'.
        upper_bound: The upper bound of integration. Can be a number or 'inf'/'-inf'.

    Returns:
        On success: {"result": <definite integral value as string>}
        On error: {"error": <error message>}

    Examples:
        >>> definite_integral("x**2", "x", "0", "1")
        {'result': '1/3'}
        >>> definite_integral("sin(x)", "x", "0", "pi")
        {'result': '2'}
        >>> definite_integral("exp(-x)", "x", "0", "inf")
        {'result': '1'}

    Notes:
        - For improper integrals, use 'inf' or '-inf' as bounds
        - The result may be symbolic if it cannot be simplified to a number
    """
    try:
        var = symbols(variable)
        expr = parse_expression(expression, variable)
        
        # Parse bounds - handle infinity
        def parse_bound(b):
            b_str = str(b).strip().lower()
            if b_str == 'inf' or b_str == '+inf' or b_str == 'infinity':
                return oo
            elif b_str == '-inf' or b_str == '-infinity':
                return -oo
            else:
                return sympify(b_str)
        
        lower = parse_bound(lower_bound)
        upper = parse_bound(upper_bound)
        
        result = sympy_integrate(expr, (var, lower, upper))
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def compute_limit(expression: str, variable: str = "x", point: str = "0", direction: str = "") -> dict:
    """
    Computes the limit of a mathematical expression as a variable approaches a point.

    Supports one-sided limits (from left or right) and limits at infinity.

    Args:
        expression: The mathematical expression as a string.
                   Examples: "sin(x)/x", "(1+1/x)**x", "1/x"
        variable: The variable for the limit. Default is "x".
        point: The point to approach. Can be a number, 'inf', or '-inf'.
        direction: Direction of approach. Use '+' for right-hand limit,
                  '-' for left-hand limit, or '' for two-sided limit.

    Returns:
        On success: {"result": <limit value as string>}
        On error: {"error": <error message>}

    Examples:
        >>> compute_limit("sin(x)/x", "x", "0")
        {'result': '1'}
        >>> compute_limit("(1+1/x)**x", "x", "inf")
        {'result': 'E'}
        >>> compute_limit("1/x", "x", "0", "+")
        {'result': 'oo'}
        >>> compute_limit("1/x", "x", "0", "-")
        {'result': '-oo'}

    Notes:
        - For infinity, use 'inf' or '-inf'
        - Direction '+' means approaching from the right (x → a⁺)
        - Direction '-' means approaching from the left (x → a⁻)
        - Empty direction '' means two-sided limit
    """
    try:
        var = symbols(variable)
        expr = parse_expression(expression, variable)
        
        # Parse the point - handle infinity
        point_str = str(point).strip().lower()
        if point_str == 'inf' or point_str == '+inf' or point_str == 'infinity':
            point_val = oo
        elif point_str == '-inf' or point_str == '-infinity':
            point_val = -oo
        else:
            point_val = sympify(point_str)
        
        # Compute limit with optional direction
        if direction == '+':
            result = sympy_limit(expr, var, point_val, '+')
        elif direction == '-':
            result = sympy_limit(expr, var, point_val, '-')
        else:
            result = sympy_limit(expr, var, point_val)
        
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def partial_fractions(expression: str, variable: str = "x") -> dict:
    """
    Performs partial fraction decomposition on a rational expression.

    Decomposes a rational function into a sum of simpler fractions.

    Args:
        expression: The rational expression to decompose as a string.
                   Examples: "1/(x**2-1)", "(x+1)/(x**2+3*x+2)", "1/(x**3-x)"
        variable: The variable in the expression. Default is "x".

    Returns:
        On success: {"result": <decomposed expression as string>}
        On error: {"error": <error message>}

    Examples:
        >>> partial_fractions("1/(x**2-1)")
        {'result': '-1/(2*(x + 1)) + 1/(2*(x - 1))'}
        >>> partial_fractions("(x+1)/(x**2+3*x+2)")
        {'result': '1/(x + 2)'}

    Notes:
        - The input must be a rational expression (polynomial/polynomial)
        - The result is fully decomposed into partial fractions
    """
    try:
        var = symbols(variable)
        expr = parse_expression(expression, variable)
        result = apart(expr, var)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def simplify_expression(expression: str) -> dict:
    """
    Simplifies a mathematical expression to its simplest form.

    Applies various algebraic simplification rules including
    combining like terms, reducing fractions, and simplifying
    trigonometric expressions.

    Args:
        expression: The expression to simplify as a string.
                   Examples: "(x**2-1)/(x-1)", "sin(x)**2 + cos(x)**2"

    Returns:
        On success: {"result": <simplified expression as string>}
        On error: {"error": <error message>}

    Examples:
        >>> simplify_expression("(x**2-1)/(x-1)")
        {'result': 'x + 1'}
        >>> simplify_expression("sin(x)**2 + cos(x)**2")
        {'result': '1'}
        >>> simplify_expression("(a+b)**2 - a**2 - 2*a*b - b**2")
        {'result': '0'}

    Notes:
        - Uses SymPy's simplify function which applies multiple strategies
        - May not always find the "simplest" form for complex expressions
    """
    try:
        expr = parse_expression(expression)
        result = sympy_simplify(expr)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


def taylor_series(expression: str, variable: str = "x", point: str = "0", order: int = 6) -> dict:
    """
    Computes the Taylor series expansion of an expression around a point.

    Args:
        expression: The expression to expand as a string.
                   Examples: "sin(x)", "exp(x)", "log(1+x)", "1/(1-x)"
        variable: The variable for the expansion. Default is "x".
        point: The point around which to expand. Default is "0" (Maclaurin series).
        order: The number of terms to compute. Default is 6.

    Returns:
        On success: {"result": <Taylor series as string>, "terms": <list of terms>}
        On error: {"error": <error message>}

    Examples:
        >>> taylor_series("sin(x)", "x", "0", 5)
        {'result': 'x - x**3/6 + O(x**5)', 'polynomial': '-x**3/6 + x', 'terms': ['-x**3/6', 'x']}
        >>> taylor_series("exp(x)", "x", "0", 4)
        {'result': '1 + x + x**2/2 + x**3/6 + O(x**4)', 'polynomial': 'x**3/6 + x**2/2 + x + 1', 'terms': ['x**3/6', 'x**2/2', 'x', '1']}

    Notes:
        - When point is 0, this is the Maclaurin series
        - The O(...) term represents the order of the remainder
    """
    try:
        var = symbols(variable)
        expr = parse_expression(expression, variable)
        point_val = sympify(str(point))
        
        # Compute Taylor series
        taylor = series(expr, var, point_val, order)
        
        # Remove the O(...) term to get just the polynomial part
        taylor_poly = taylor.removeO()
        
        # Get individual terms
        terms = []
        if taylor_poly.is_Add:
            terms = [str(term) for term in taylor_poly.as_ordered_terms()]
        else:
            terms = [str(taylor_poly)]
        
        return {
            "result": str(taylor),
            "polynomial": str(taylor_poly),
            "terms": terms
        }
    except Exception as e:
        return {"error": str(e)}


def solve_system(equations: List[str], variables: List[str] = None) -> dict:
    """
    Solves a system of equations for multiple variables with verification.

    Args:
        equations: List of equations as strings. Each equation should contain '='.
                  Examples: ["x + y = 5", "x - y = 1"]
        variables: Optional list of variable names to solve for.
                  If not provided, variables are auto-detected.

    Returns:
        On success: {
            "solutions": <list of solution dicts>,
            "equations": <list of original equations>,
            "num_equations": <count>,
            "num_variables": <count>,
            "variables": <list of variable names>,
            "num_solutions": <count of solutions>,
            "verification": [{"equation": ..., "result": ...}],
            "method": <solving method>
        }
        On error: {"error": <error message>}

    Examples:
        >>> solve_system(["x + y = 5", "x - y = 1"])
        {'solutions': [{'x': 3, 'y': 2}], 'num_solutions': 1, 'verification': [...], ...}
        >>> solve_system(["x + y + z = 6", "x - y = 0", "y + z = 4"])
        {'solutions': [{'x': 2, 'y': 2, 'z': 2}], 'num_equations': 3, ...}

    Notes:
        - Equations must contain exactly one '=' sign
        - Supports linear and nonlinear systems
        - For systems with no solution, returns empty solutions list
        - Each solution is verified by substitution into all equations
    """
    try:
        # Parse equations
        sympy_eqs = []
        detected_vars = set()
        normalized_eqs = []
        
        for eq_str in equations:
            eq_str = normalize_expression(eq_str)
            normalized_eqs.append(eq_str)
            parts = eq_str.split('=')
            if len(parts) != 2:
                return {"error": f"Invalid equation format: {eq_str}. Each equation must contain exactly one '=' sign."}
            
            left = sympify(parts[0].strip())
            right = sympify(parts[1].strip())
            sympy_eqs.append(Eq(left, right))
            
            # Detect variables
            detected_vars.update(left.free_symbols)
            detected_vars.update(right.free_symbols)
        
        # Use provided variables or detected ones
        if variables:
            solve_vars = [symbols(v) for v in variables]
            var_names = variables
        else:
            solve_vars = list(detected_vars)
            var_names = [str(v) for v in solve_vars]
        
        # Solve the system
        solutions = solve(sympy_eqs, solve_vars, dict=True)
        
        if not solutions:
            # Try without dict=True for systems that return tuples
            solutions = solve(sympy_eqs, solve_vars)
        
        # Convert solutions to list of dicts
        solution_list = []
        if isinstance(solutions, list):
            for sol in solutions:
                if isinstance(sol, dict):
                    solution_list.append({str(k): str(v) for k, v in sol.items()})
                else:
                    # Handle tuple solutions
                    solution_list.append({var_names[i]: str(solutions[i]) for i in range(len(var_names))})
        elif isinstance(solutions, dict):
            solution_list = [solutions]
        else:
            solution_list = []
        
        # Verify each solution by substituting back into equations
        verification = []
        for sol_dict in solution_list:
            for eq_idx, eq in enumerate(sympy_eqs):
                try:
                    # Substitute solution into equation
                    subs_dict = {symbols(k): sympify(v) for k, v in sol_dict.items()}
                    result = eq.subs(subs_dict)
                    is_valid = result == True or simplify(result) == True
                    verification.append({
                        "equation": normalized_eqs[eq_idx],
                        "solution": sol_dict,
                        "result": str(result),
                        "valid": is_valid
                    })
                except:
                    verification.append({
                        "equation": normalized_eqs[eq_idx],
                        "solution": sol_dict,
                        "valid": None
                    })
        
        # Determine solving method
        if len(sympy_eqs) == len(solve_vars):
            method = "Square system (n equations, n variables)"
        elif len(sympy_eqs) > len(solve_vars):
            method = "Overdetermined system (more equations than variables)"
        else:
            method = "Underdetermined system (fewer equations than variables)"
        
        return {
            "solutions": solution_list,
            "equations": normalized_eqs,
            "num_equations": len(sympy_eqs),
            "num_variables": len(solve_vars),
            "variables": var_names,
            "num_solutions": len(solution_list),
            "verification": verification,
            "method": method
        }
    except Exception as e:
        return {"error": str(e)}


def mean(data: List[float]) -> dict:
    """
    Computes the mean (average) of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {
            "result": <mean value>,
            "data": <input data>,
            "count": <number of values>,
            "sum": <sum of all values>,
            "formula": <calculation breakdown>,
            "interpretation": <description>
        }
        On error: {"error": <error message>}

    Examples:
        >>> mean([1, 2, 3, 4])
        {'result': 2.5, 'sum': 10, 'count': 4, ...}
        >>> mean([10, 20, 30])
        {'result': 20.0, 'sum': 60, 'count': 3, ...}
    """
    try:
        result = float(np.mean(data))
        total_sum = float(np.sum(data))
        count = len(data)
        formula = f"sum / count = {total_sum} / {count}"
        interpretation = "Average of the values. Best used with normally distributed data without outliers."
        
        return {
            "result": result,
            "data": data,
            "count": count,
            "sum": total_sum,
            "formula": formula,
            "interpretation": interpretation
        }
    except Exception as e:
        return {"error": str(e)}


def variance(data: List[float]) -> dict:
    """
    Computes the variance of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <variance value>}
        On error: {"error": <error message>}

    Examples:
        >>> variance([1, 2, 3, 4])
        {'result': 1.25}
    """
    try:
        result = float(np.var(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def standard_deviation(data: List[float]) -> dict:
    """
    Computes the standard deviation of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <standard deviation value>}
        On error: {"error": <error message>}

    Examples:
        >>> standard_deviation([1, 2, 3, 4])
        {'result': 1.118033988749895}
    """
    try:
        result = float(np.std(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def median(data: List[float]) -> dict:
    """
    Computes the median of a list of numbers.

    The median is the middle value when data is sorted. Better than mean for data with outliers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {
            "result": <median value>,
            "data": <input data>,
            "count": <number of values>,
            "sorted_data": <sorted data>,
            "position": <description of position>,
            "interpretation": <description>
        }
        On error: {"error": <error message>}

    Examples:
        >>> median([1, 2, 3, 4])
        {'result': 2.5, 'position': 'between indices 1 and 2', ...}
        >>> median([1, 3, 5, 7, 9])
        {'result': 5.0, 'position': 'middle value (index 2)', ...}
    """
    try:
        result = float(np.median(data))
        count = len(data)
        sorted_data = sorted(data)
        
        # Determine position
        if count % 2 == 1:
            mid_idx = count // 2
            position = f"middle value (index {mid_idx} of {count})"
        else:
            mid_idx1 = count // 2 - 1
            mid_idx2 = count // 2
            position = f"average of indices {mid_idx1} and {mid_idx2}"
        
        interpretation = "Middle value when data is sorted. Better than mean when data has outliers."
        
        return {
            "result": result,
            "data": data,
            "count": count,
            "sorted_data": sorted_data,
            "position": position,
            "interpretation": interpretation
        }
    except Exception as e:
        return {"error": str(e)}


def mode(data: List[float]) -> dict:
    """
    Computes the mode of a list of numbers.

    The mode is the value that appears most frequently in the dataset.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {
            "result": <mode value>,
            "data": <input data>,
            "count": <total count>,
            "frequency": <how many times mode appears>,
            "percentage": <percentage of total>,
            "interpretation": <description>
        }
        On error: {"error": <error message>}

    Examples:
        >>> mode([1, 2, 2, 3, 3, 3, 4])
        {'result': 3.0, 'frequency': 3, 'percentage': 42.86, ...}
        >>> mode([1, 1, 2, 2])
        {'result': 1.0, 'frequency': 2, 'percentage': 50.0, ...}
        >>> mode([])
        {'error': 'Cannot compute mode of empty array'}
    """
    try:
        if not data:
            return {"error": "Cannot compute mode of empty array"}
        
        # Adjusted for newer SciPy versions
        mode_result = stats.mode(data, keepdims=False)
        mode_value = float(mode_result.mode)
        frequency = int(mode_result.count)
        count = len(data)
        percentage = (frequency / count) * 100
        
        interpretation = f"The value {mode_value} appears {frequency} times ({percentage:.1f}% of data). Most frequent value."
        
        return {
            "result": mode_value,
            "data": data,
            "count": count,
            "frequency": frequency,
            "percentage": round(percentage, 2),
            "interpretation": interpretation
        }
    except Exception as e:
        return {"error": str(e)}


def correlation_coefficient(data_x: List[float], data_y: List[float]) -> dict:
    """
    Computes the Pearson correlation coefficient between two lists of numbers.

    Measures the strength and direction of the linear relationship between two variables.

    Args:
        data_x: The first list of numerical values.
        data_y: The second list of numerical values.

    Returns:
        On success: {
            "result": <correlation coefficient>,
            "data_x": <input data>,
            "data_y": <input data>,
            "count": <number of data points>,
            "strength": <strength descriptor>,
            "direction": <positive or negative>,
            "interpretation": <human-readable interpretation>,
            "equation_note": <note about use in regression>
        }
        On error: {"error": <error message>}

    Examples:
        >>> correlation_coefficient([1, 2, 3], [4, 5, 6])
        {'result': 1.0, 'strength': 'Perfect', 'direction': 'positive', ...}
        >>> correlation_coefficient([1, 2, 3], [6, 4, 2])
        {'result': -1.0, 'strength': 'Perfect', 'direction': 'negative', ...}
    """
    try:
        result = float(np.corrcoef(data_x, data_y)[0, 1])
        count = len(data_x)
        
        # Determine strength
        abs_r = abs(result)
        if abs_r >= 0.9:
            strength = "Very strong"
        elif abs_r >= 0.7:
            strength = "Strong"
        elif abs_r >= 0.5:
            strength = "Moderate"
        elif abs_r >= 0.3:
            strength = "Weak"
        elif abs_r >= 0.1:
            strength = "Very weak"
        else:
            strength = "Negligible"
        
        # Determine direction
        if result > 0:
            direction = "positive"
            dir_text = "As X increases, Y tends to increase"
        elif result < 0:
            direction = "negative"
            dir_text = "As X increases, Y tends to decrease"
        else:
            direction = "none"
            dir_text = "No linear relationship"
        
        # Build interpretation 
        interpretation = f"{strength} {direction} correlation: {dir_text}"
        
        # Note about regression
        equation_note = f"R² = {result**2:.4f} (coefficient of determination: {result**2*100:.2f}% of variance explained)"
        
        return {
            "result": result,
            "data_x": data_x,
            "data_y": data_y,
            "count": count,
            "strength": strength,
            "direction": direction,
            "interpretation": interpretation,
            "r_squared_equivalent": round(result**2, 4),
            "equation_note": equation_note
        }
    except Exception as e:
        return {"error": str(e)}


def linear_regression(data: List[Tuple[float, float]]) -> dict:
    """
    Performs linear regression on a set of points and returns the slope and intercept.

    Args:
        data: A list of tuples, where each tuple contains (x, y) coordinates.

    Returns:
        On success: {
            "slope": <slope value>, 
            "intercept": <intercept value>,
            "equation": <equation string>,
            "r_value": <correlation coefficient>,
            "r_squared": <coefficient of determination>,
            "p_value": <statistical significance>,
            "std_err": <standard error of slope>,
            "fit_quality": <qualitative assessment>
        }
        On error: {"error": <error message>}

    Examples:
        >>> linear_regression([(1, 2), (2, 3), (3, 5)])
        {'slope': 1.5, 'intercept': 0.33, 'r_squared': 0.95, ...}
    """
    try:
        x = np.array([point[0] for point in data])
        y = np.array([point[1] for point in data])
        result = stats.linregress(x, y)
        
        slope = float(result.slope)
        intercept = float(result.intercept)
        r_value = float(result.rvalue)
        r_squared = float(result.rvalue ** 2)
        p_value = float(result.pvalue)
        std_err = float(result.stderr)
        
        # Determine fit quality
        if r_squared > 0.99:
            fit_quality = "Excellent (R² > 0.99)"
        elif r_squared > 0.95:
            fit_quality = "Very Good (R² > 0.95)"
        elif r_squared > 0.80:
            fit_quality = "Good (R² > 0.80)"
        elif r_squared > 0.50:
            fit_quality = "Moderate (R² > 0.50)"
        else:
            fit_quality = "Poor (R² ≤ 0.50)"
        
        return {
            "slope": slope,
            "intercept": intercept,
            "equation": f"y = {slope:.6f}x + {intercept:.6f}",
            "r_value": r_value,
            "r_squared": r_squared,
            "p_value": p_value,
            "std_err": std_err,
            "fit_quality": fit_quality,
            "data_points": len(data),
            "interpretation": f"Strong positive correlation" if r_value > 0.7 else f"Weak correlation" if r_value < 0.3 else f"Moderate correlation"
        }
    except Exception as e:
        return {"error": str(e)}


def confidence_interval(data: List[float], confidence: float = 0.95) -> dict:
    """
    Computes the confidence interval for the mean of a dataset.

    Args:
        data: A list of numerical values.
        confidence: The confidence level (default is 0.95).

    Returns:
        On success: {"confidence_interval": <(lower_bound, upper_bound)>}
        On error: {"error": <error message>}

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducible results
        >>> confidence_interval([1, 2, 3, 4])
        {'confidence_interval': (0.445739743239121, 4.5542602567608785)}
    """
    try:
        mean_value = np.mean(data)
        sem = stats.sem(data)  # Standard error of the mean
        margin_of_error = sem * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
        return {
            "confidence_interval": (
                float(mean_value - margin_of_error),
                float(mean_value + margin_of_error),
            )
        }
    except Exception as e:
        return {"error": str(e)}


def matrix_addition(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> dict:
    """
    Adds two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        On success: {"result": <resulting matrix>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_addition([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        {'result': [[6, 8], [10, 12]]}
    """
    try:
        result = np.add(matrix_a, matrix_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def matrix_multiplication(
    matrix_a: List[List[float]], matrix_b: List[List[float]]
) -> dict:
    """
    Multiplies two matrices.

    Args:
        matrix_a: The first matrix as a list of lists.
        matrix_b: The second matrix as a list of lists.

    Returns:
        On success: {"result": <resulting matrix>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_multiplication([[1, 2], [3, 4]], [[5, 6], [7, 8]])
        {'result': [[19, 22], [43, 50]]}
    """
    try:
        result = np.dot(matrix_a, matrix_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def matrix_transpose(matrix: List[List[float]]) -> dict:
    """
    Transposes a matrix.

    Args:
        matrix: The matrix to transpose as a list of lists.

    Returns:
        On success: {"result": <transposed matrix>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_transpose([[1, 2], [3, 4]])
        {'result': [[1, 3], [2, 4]]}
    """
    try:
        result = np.transpose(matrix).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def matrix_determinant(matrix: List[List[float]]) -> dict:
    """
    Multiplies two matrices.

    Args:
        matrix: The first vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_determinant([[1, 2], [3, 4]])
        {'result': -2.0}
    """
    try:
        result = np.linalg.det(matrix)
        return {"result": round(float(result), 10)}
    except Exception as e:
        return {"error": str(e)}


def matrix_eigenvalues(matrix: List[List[float]]) -> dict:
    """
    Computes the eigenvalues of a square matrix.

    Args:
        matrix: The matrix as a list of lists. Must be a square matrix.

    Returns:
        On success: {"result": <list of eigenvalues>, "explanation": <description>}
        On error: {"error": <error message>}

    Examples:
        >>> matrix_eigenvalues([[1, 2], [3, 4]])
        {'result': [-0.3722813232690143, 5.372281323269014], 'explanation': 'Calculated 2 eigenvalues'}
        >>> matrix_eigenvalues([[2, 0], [0, 3]])
        {'result': [2.0, 3.0], 'explanation': 'Calculated 2 eigenvalues'}
    """
    try:
        mat = np.array(matrix, dtype=float)
        
        # Check if matrix is square
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            return {"error": "Eigenvalues require a square matrix (same number of rows and columns)"}
        
        eigenvalues = np.linalg.eigvals(mat)
        
        # Handle complex eigenvalues
        if np.iscomplexobj(eigenvalues):
            # Check if imaginary parts are negligible (essentially real)
            if np.allclose(eigenvalues.imag, 0):
                result = [float(ev.real) for ev in eigenvalues]
            else:
                result = [{"real": float(ev.real), "imag": float(ev.imag)} for ev in eigenvalues]
        else:
            result = [float(ev) for ev in eigenvalues]
        
        return {
            "result": result,
            "explanation": f"Calculated {len(eigenvalues)} eigenvalues"
        }
    except Exception as e:
        return {"error": str(e)}


def vector_dot_product(vector_a: tuple[float], vector_b: tuple[float]) -> dict:
    """
    Multiplies two matrices.

    Args:
        vector_a: The first vector as a list of lists.
        vector_b: The second vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}

    Examples:
        >>> vector_dot_product([1, 2], [7, 8])
        {'result': 23}
    """
    try:
        result = np.dot(vector_a, vector_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def vector_cross_product(vector_a: tuple[float], vector_b: tuple[float]) -> dict:
    """
    Multiplies two matrices.

    Args:
        vector_a: The first vector as a list of lists.
        vector_b: The second vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}

    Examples:
        >>> vector_cross_product([1, 2, 3], [4, 5, 6])
        {'result': [-3, 6, -3]}
    """
    try:
        result = np.cross(vector_a, vector_b).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def vector_magnitude(vector: tuple[float]) -> dict:
    """
    Multiplies two matrices.

    Args:
        vector: The first vector as a list of lists.

    Returns:
        On success: {"result": <resulting vector>}
        On error: {"error": <error message>}

    Examples:
        >>> vector_magnitude([1, 2, 3])
        {'result': 3.7416573867739413}
    """
    try:
        result = np.linalg.norm(vector).tolist()
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def plot_function(
    expression: str, start: int = -10, end: int = 10, step: int = 100
) -> dict:
    """
    Plots a graph of y = f(x).

    Args:
        x: The expression of function x as a string.

    Returns:
        On success: {"result": "Plot generated successfully."}
        On error: {"error": <error message>}

    Examples:
        >>> plot_function("x**2")
        {'result': 'Plot generated successfully.'}

    Notes:
        - Use 'x' as the variable (e.g., x**2, not x²)
        - Multiplication must be explicitly indicated with * (e.g., 2*x, not 2x)
        - Powers are represented with ** (e.g., x**2, not x^2)
    """
    x = sp.Symbol("x")
    try:
        expression = parse_expression(expression)
        f = sp.lambdify(x, expression, "numpy")
        x_values = np.linspace(start, end, step)
        y_values = f(x_values)
        fig, ax = plt.subplots()
        # Create quadrant graph
        ax.spines["left"].set_position("center")
        ax.spines["bottom"].set_position("center")
        ax.spines["right"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        ax.plot(x_values, y_values)
        ax.set_xlabel("x", loc="right")
        ax.set_ylabel("f(x)", loc="top")
        ax.set_title(f"Graph of ${sp.latex(expression)}$")
        ax.grid(True)
        plt.show()
        return {"result": "Plot generated successfully."}
    except Exception as e:
        return {"error": str(e)}


def summation(expression: str, start: int = 0, end: int = 10) -> dict:
    """
    Calculates the summation of a function from start to end.

    Args:
        expression: The expression of function x as a string.
        start: The starting value of the summation.
        end: The ending value of the summation.

    Returns:
        On success: {"result": <resulting summation>}
        On error: {"error": <error message>}

    Examples:
        >>> summation("x**2", 0, 10)
        {'result': 385}
    """
    try:
        x = sp.Symbol("x")
        expr = parse_expression(expression)
        summation = sp.Sum(expr, (x, start, end))
        result = summation.doit()
        return {"result": int(result) if result.is_integer else float(result)}
    except Exception as e:
        return {"error": str(e)}


def expand(expression: str) -> dict:
    """
    Expands an algebraic expression by removing parentheses and combining terms.

    Args:
        expression: The expression to expand as a string.

    Returns:
        On success: {
            "result": <expanded expression>,
            "input": <original expression>,
            "method": <expansion method>,
            "interpretation": <description>,
            "note": <helpful note about expansion>
        }
        On error: {"error": <error message>}

    Examples:
        >>> expand("(x + 1)**2")
        {'result': 'x**2 + 2*x + 1', 'method': 'Binomial expansion', ...}
        >>> expand("(x + 1)*(x - 1)")
        {'result': 'x**2 - 1', 'method': 'Difference of squares', ...}
    """
    try:
        x = sp.Symbol("x")
        parsed_expr = parse_expression(expression)
        expanded_expression = sp.expand(parsed_expr)
        
        # Determine the expansion method
        expr_str = str(parsed_expr)
        if "**2" in expr_str and "+" in expr_str:
            method = "Binomial expansion: (a+b)² = a² + 2ab + b²"
        elif "**3" in expr_str and "+" in expr_str:
            method = "Trinomial expansion: (a+b)³ = a³ + 3a²b + 3ab² + b³"
        elif "*" in expr_str and "(" in expr_str:
            if "-" in expr_str and ")" in expr_str:
                method = "Product expansion (difference of squares)"
            else:
                method = "Distributive property: a(b+c) = ab + ac"
        else:
            method = "Algebraic expansion (SymPy)"
        
        interpretation = "Removes parentheses and combines like terms into standard form."
        note = "Expanded form is useful for finding roots and analyzing polynomial behavior."
        
        return {
            "result": str(expanded_expression),
            "input": expression,
            "method": method,
            "interpretation": interpretation,
            "note": note
        }
    except Exception as e:
        return {"error": str(e)}


def factorize(expression: str) -> dict:
    """
    Factorizes an algebraic expression by finding common factors and patterns.

    Args:
        expression: The expression to factorize as a string.

    Returns:
        On success: {
            "result": <factored expression>,
            "input": <original expression>,
            "method": <factorization method>,
            "interpretation": <description>,
            "note": <helpful note about factorization>
        }
        On error: {"error": <error message>}

    Examples:
        >>> factorize("x**2 + 2*x + 1")
        {'result': '(x + 1)**2', 'method': 'Perfect square trinomial', ...}
        >>> factorize("x**2 - 1")
        {'result': '(x - 1)*(x + 1)', 'method': 'Difference of squares', ...}
    """
    try:
        x = sp.Symbol("x")
        parsed_expr = parse_expression(expression)
        factored_expression = sp.factor(parsed_expr)
        
        # Determine the factorization method
        expr_str = str(parsed_expr)
        result_str = str(factored_expression)
        
        if "**2" in result_str:
            method = "Perfect square trinomial: a² + 2ab + b² = (a+b)²"
        elif "+" in expr_str and "-" in expr_str and "**2" in expr_str:
            method = "Difference of squares: a² - b² = (a-b)(a+b)"
        elif "-" in expr_str and "**2" not in expr_str:
            method = "Difference of squares or linear factors"
        else:
            method = "Polynomial factorization (SymPy)"
        
        interpretation = "Writes expression as product of simpler factors."
        note = "Factored form is useful for finding zeros/roots and solving equations."
        
        return {
            "result": str(factored_expression),
            "input": expression,
            "method": method,
            "interpretation": interpretation,
            "note": note
        }
    except Exception as e:
        return {"error": str(e)}

