from mcp.server.fastmcp import FastMCP
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
app = FastMCP(
    name="Mathematical Calculator",
    instructions="A server for complex mathematical calculations",
    dependencies=["numpy", "scipy", "sympy", "matplotlib"],
)

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


def normalize_expression(expression: str, variable: str = "x") -> str:
    """
    Normalize common user math input into SymPy-friendly syntax.

    - Replace caret power (^) with Python power (**)
    - Insert explicit multiplication where commonly omitted:
      3x -> 3*x, 2(x+1) -> 2*(x+1), (x+1)(x+2) -> (x+1)*(x+2), x(x+1) -> x*(x+1)

    Only handles implicit multiplication involving the primary variable and parentheses
    to avoid interfering with function names like sin(x).
    """
    s = expression or ""
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


@app.tool()
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


@app.tool()
def solve_equation(equation: str) -> dict:
    """
    Solves an algebraic equation for x and returns all solutions.

    The equation must contain exactly one equality sign (=) and use a
    variable x. Can solve polynomial, trigonometric, and other equations
    supported by SymPy.

    Args:
        equation: The equation to solve as a string.
                 Format: '<left side> = <right side>'
                 Examples: "x**2 - 5*x + 6 = 0", "sin(x) = 0.5", "2*x + 3 = 7"

    Returns:
        On success: {"solutions": <list of solutions as string>}
        On error: {"error": <error message>}

    Examples:
        >>> solve_equation("x**2 - 5*x + 6 = 0")
        {'solutions': '[2, 3]'}
        >>> solve_equation("2*x + 3 = 7")
        {'solutions': '[2]'}
        >>> solve_equation("x = 0")
        {'solutions': '[0]'}

    Notes:
        - Use 'x' as the variable (e.g., x**2, not x²)
        - Multiplication must be explicitly indicated with * (e.g., 2*x, not 2x)
        - Powers are represented with ** (e.g., x**2, not x^2)
    """
    try:
        x = symbols("x")
        # Split the equation into left and right sides
        parts = equation.split("=")
        if len(parts) != 2:
            return {"error": "Equation must contain an '=' sign"}

        left = sympify(normalize_expression(parts[0].strip()))
        right = sympify(normalize_expression(parts[1].strip()))

        # Solve the equation
        solutions = solve(left - right, x)
        return {"solutions": str(solutions)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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
        On success: {"result": <derivative as string>}
        On error: {"error": <error message>}

    Examples:
        >>> differentiate("x**2")
        {'result': '2*x'}
        >>> differentiate("sin(x)")
        {'result': 'cos(x)'}
        >>> differentiate("x*y", "y")
        {'result': 'x'}
        >>> differentiate("exp(x)")
        {'result': 'exp(x)'}

    Notes:
        - Use mathematical notation with explicit operators (* for multiplication)
        - Powers are represented with ** (e.g., x**2, not x^2)
        - For trigonometric functions, use sin(x), cos(x), etc.
        - Only support for one variable at a time (implicit differentiation not supported)
    """
    try:
        var = symbols(variable)
        expr = sympify(normalize_expression(expression, variable=variable))
        result = diff(expr, var)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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
        On success: {"result": <integral as string>}
        On error: {"error": <error message>}

    Examples:
        >>> integrate("x**2")
        {'result': 'x**3/3'}
        >>> integrate("sin(x)")
        {'result': '-cos(x)'}
        >>> integrate("exp(x)")
        {'result': 'exp(x)'}
        >>> integrate("1/x")
        {'result': 'log(x)'}
        >>> integrate("x*y", "y")
        {'result': 'x*y**2/2'}

    Notes:
        - The result is the indefinite integral without the constant of integration
        - Complex expressions may be returned in simplified form
    """
    try:
        var = symbols(variable)
        expr = sympify(normalize_expression(expression, variable=variable))
        result = sympy_integrate(expr, var)  # Use sympy_integrate instead of integrate
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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
        expr = sympify(normalize_expression(expression, variable=variable))
        
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


@app.tool()
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
        expr = sympify(normalize_expression(expression, variable=variable))
        
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


@app.tool()
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
        {'result': '2/(x + 2) - 1/(x + 1)'}

    Notes:
        - The input must be a rational expression (polynomial/polynomial)
        - The result is fully decomposed into partial fractions
    """
    try:
        var = symbols(variable)
        expr = sympify(normalize_expression(expression, variable=variable))
        result = apart(expr, var)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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
        expr = sympify(normalize_expression(expression))
        result = sympy_simplify(expr)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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
        {'result': 'x - x**3/6 + x**5/120 + O(x**6)', 'terms': ['x', '-x**3/6', 'x**5/120']}
        >>> taylor_series("exp(x)", "x", "0", 4)
        {'result': '1 + x + x**2/2 + x**3/6 + O(x**4)', 'terms': ['1', 'x', 'x**2/2', 'x**3/6']}

    Notes:
        - When point is 0, this is the Maclaurin series
        - The O(...) term represents the order of the remainder
    """
    try:
        var = symbols(variable)
        expr = sympify(normalize_expression(expression, variable=variable))
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


@app.tool()
def solve_system(equations: List[str], variables: List[str] = None) -> dict:
    """
    Solves a system of equations for multiple variables.

    Args:
        equations: List of equations as strings. Each equation should contain '='.
                  Examples: ["x + y = 5", "x - y = 1"]
        variables: Optional list of variable names to solve for.
                  If not provided, variables are auto-detected.

    Returns:
        On success: {"solutions": <dict or list of solutions>}
        On error: {"error": <error message>}

    Examples:
        >>> solve_system(["x + y = 5", "x - y = 1"])
        {'solutions': '{x: 3, y: 2}'}
        >>> solve_system(["x + y + z = 6", "x - y = 0", "y + z = 4"])
        {'solutions': '{x: 2, y: 2, z: 2}'}
        >>> solve_system(["x**2 + y**2 = 25", "x = y"])
        {'solutions': '[(-5*sqrt(2)/2, -5*sqrt(2)/2), (5*sqrt(2)/2, 5*sqrt(2)/2)]'}

    Notes:
        - Equations must contain exactly one '=' sign
        - Supports linear and nonlinear systems
        - For systems with no solution, returns empty result
    """
    try:
        # Parse equations
        sympy_eqs = []
        detected_vars = set()
        
        for eq_str in equations:
            eq_str = normalize_expression(eq_str)
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
        else:
            solve_vars = list(detected_vars)
        
        # Solve the system
        solutions = solve(sympy_eqs, solve_vars, dict=True)
        
        if not solutions:
            # Try without dict=True for systems that return tuples
            solutions = solve(sympy_eqs, solve_vars)
        
        return {"solutions": str(solutions)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def mean(data: List[float]) -> dict:
    """
    Computes the mean of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <mean value>}
        On error: {"error": <error message>}

    Examples:
        >>> mean([1, 2, 3, 4])
        {'result': 2.5}
        >>> mean([10, 20, 30])
        {'result': 20.0}
    """
    try:
        result = float(np.mean(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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


@app.tool()
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


@app.tool()
def median(data: List[float]) -> dict:
    """
    Computes the median of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <median value>}
        On error: {"error": <error message>}

    Examples:
        >>> median([1, 2, 3, 4])
        {'result': 2.5}
    """
    try:
        result = float(np.median(data))
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def mode(data: List[float]) -> dict:
    """
    Computes the mode of a list of numbers.

    Args:
        data: A list of numerical values.

    Returns:
        On success: {"result": <mode value>}
        On error: {"error": <error message>}

    Examples:
        >>> mode([1, 2, 2, 3])
        {'result': 2.0}
        >>> mode([1, 1, 2, 2])
        {'result': 1.0}
        >>> mode([])
        {'error': 'Cannot compute mode of empty array'}
    """
    try:
        if not data:
            return {"error": "Cannot compute mode of empty array"}
        # Adjusted for newer SciPy versions
        mode_result = stats.mode(data, keepdims=False)
        return {"result": float(mode_result.mode)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def correlation_coefficient(data_x: List[float], data_y: List[float]) -> dict:
    """
    Computes the Pearson correlation coefficient between two lists of numbers.

    Args:
        data_x: The first list of numerical values.
        data_y: The second list of numerical values.

    Returns:
        On success: {"result": <correlation coefficient>}
        On error: {"error": <error message>}

    Examples:
        >>> correlation_coefficient([1, 2, 3], [4, 5, 6])
        {'result': 1.0}
    """
    try:
        result = np.corrcoef(data_x, data_y)[0, 1]
        return {"result": float(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def linear_regression(data: List[Tuple[float, float]]) -> dict:
    """
    Performs linear regression on a set of points and returns the slope and intercept.

    Args:
        data: A list of tuples, where each tuple contains (x, y) coordinates.

    Returns:
        On success: {"slope": <slope value>, "intercept": <intercept value>}
        On error: {"error": <error message>}

    Examples:
        >>> linear_regression([(1, 2), (2, 3), (3, 5)])
        {'slope': 1.5, 'intercept': 0.3333333333333335}
    """
    try:
        x = np.array([point[0] for point in data])
        y = np.array([point[1] for point in data])
        slope, intercept, _, _, _ = stats.linregress(x, y)
        return {"slope": float(slope), "intercept": float(intercept)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
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


@app.tool()
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


@app.tool()
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


@app.tool()
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


@app.tool()
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


@app.tool()
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
        {'result': [5.372281323269014, -0.3722813232690143], 'explanation': 'Calculated 2 eigenvalues'}
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


@app.tool()
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


@app.tool()
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


@app.tool()
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


@app.tool()
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
        expression = sp.sympify(normalize_expression(expression))
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


@app.tool()
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
        expr = sp.sympify(normalize_expression(expression))
        summation = sp.Sum(expr, (x, start, end))
        result = summation.doit()
        return {"result": int(result) if result.is_integer else float(result)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def expand(expression: str) -> dict:
    """
    Expands an expression.

    Args:
        expression: The expression to expand as a string.

    Returns:
        On success: {"result": <expanded expression>}
        On error: {"error": <error message>}

    Examples:
        >>> expand("(x + 1)**2")
        {'result': 'x**2 + 2*x + 1'}
    """
    try:
        x = sp.Symbol("x")
        expanded_expression = sp.expand(normalize_expression(expression))
        return {"result": str(expanded_expression)}
    except Exception as e:
        return {"error": str(e)}


@app.tool()
def factorize(expression: str) -> dict:
    """
    Factorizes an expression.

    Args:
        expression: The expression to factorize as a string.

    Returns:
        On success: {"result": <factored expression>}
        On error: {"error": <error message>}

    Examples:
        >>> factorize("x**2 + 2*x + 1")
        {'result': '(x + 1)**2'}
    """
    try:
        x = sp.Symbol("x")
        factored_expression = sp.factor(normalize_expression(expression))
        return {"result": str(factored_expression)}
    except Exception as e:
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="Mathematical Calculator MCP Server")
    parser.add_argument("--stdio", action="store_true", help="Use STDIO transport instead of SSE")
    args = parser.parse_args()
    
    transport = "stdio" if args.stdio else TRANSPORT
    app.run(transport=transport)

if __name__ == "__main__":
    main()
