"""
test.py -- Comprehensive test suite for all 28 calculator MCP tools.

Tests each math tool function directly (no LLM calls involved).
Each test calls the function, checks that it returns a valid result
(no "error" key), and validates the output against expected values.

Usage:
    python test.py
"""

import sys
import math
import traceback

from app.services.math_service import (
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
    matrix_eigenvalues,
    vector_dot_product,
    vector_cross_product,
    vector_magnitude,
    summation,
    expand,
    factorize,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0
ERRORS = []


def assert_no_error(result: dict, tool_name: str):
    """Raise if the tool returned an error dict instead of a valid result."""
    if "error" in result:
        raise AssertionError(f"Tool returned error: {result['error']}")


def assert_close(actual, expected, tol=1e-6, msg=""):
    """Assert two numeric values are within tolerance."""
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"Expected {expected}, got {actual} (diff={abs(actual - expected)}) {msg}"
        )


def assert_equal(actual, expected, msg=""):
    """Assert two values are exactly equal."""
    if actual != expected:
        raise AssertionError(f"Expected {expected!r}, got {actual!r} {msg}")


def run_test(name: str, func):
    """Run a single test, track pass/fail, and print result."""
    global PASSED, FAILED
    try:
        func()
        PASSED += 1
        print(f"  [PASS] {name}")
    except Exception as exc:
        FAILED += 1
        ERRORS.append((name, exc))
        print(f"  [FAIL] {name} -- {exc}")
        traceback.print_exc(limit=2)
        print()


# ── 1. calculate ─────────────────────────────────────────────────────────────

def test_calculate_basic_arithmetic():
    """Basic arithmetic: 2 + 3 * 4 = 14"""
    res = calculate("2 + 3 * 4")
    assert_no_error(res, "calculate")
    assert_equal(res["result"], 14)


def test_calculate_trig():
    """Trigonometric function: sin(pi/2) = 1.0"""
    res = calculate("sin(pi/2)")
    assert_no_error(res, "calculate")
    assert_close(res["result"], 1.0)


def test_calculate_sqrt():
    """Square root: sqrt(144) = 12.0"""
    res = calculate("sqrt(144)")
    assert_no_error(res, "calculate")
    assert_close(res["result"], 12.0)


def test_calculate_power():
    """Power operator: 2**10 = 1024"""
    res = calculate("2**10")
    assert_no_error(res, "calculate")
    assert_equal(res["result"], 1024)


def test_calculate_log():
    """Natural log: log(e) = 1.0"""
    res = calculate("log(e)")
    assert_no_error(res, "calculate")
    assert_close(res["result"], 1.0)


# ── 2. solve_equation ───────────────────────────────────────────────────────

def test_solve_equation_linear():
    """Linear equation: 2*x + 3 = 7 => x = 2"""
    res = solve_equation("2*x + 3 = 7")
    assert_no_error(res, "solve_equation")
    assert_equal(res["solutions"], "[2]")


def test_solve_equation_quadratic():
    """Quadratic equation: x**2 - 5*x + 6 = 0 => x = 2, 3"""
    res = solve_equation("x**2 - 5*x + 6 = 0")
    assert_no_error(res, "solve_equation")
    solutions = res["solutions"]
    assert "2" in solutions and "3" in solutions


# ── 3. differentiate ────────────────────────────────────────────────────────

def test_differentiate_polynomial():
    """d/dx(x**3) = 3*x**2"""
    res = differentiate("x**3")
    assert_no_error(res, "differentiate")
    assert_equal(res["result"], "3*x**2")


def test_differentiate_trig():
    """d/dx(sin(x)) = cos(x)"""
    res = differentiate("sin(x)")
    assert_no_error(res, "differentiate")
    assert_equal(res["result"], "cos(x)")


# ── 4. integrate ─────────────────────────────────────────────────────────────

def test_integrate_polynomial():
    """∫ x**2 dx = x**3/3"""
    res = integrate("x**2")
    assert_no_error(res, "integrate")
    assert_equal(res["result"], "x**3/3")


def test_integrate_trig():
    """∫ cos(x) dx = sin(x)"""
    res = integrate("cos(x)")
    assert_no_error(res, "integrate")
    assert_equal(res["result"], "sin(x)")


# ── 5. definite_integral ────────────────────────────────────────────────────

def test_definite_integral_polynomial():
    """∫₀¹ x**2 dx = 1/3"""
    res = definite_integral("x**2", "x", "0", "1")
    assert_no_error(res, "definite_integral")
    assert_equal(res["result"], "1/3")


def test_definite_integral_trig():
    """∫₀ᵖⁱ sin(x) dx = 2"""
    res = definite_integral("sin(x)", "x", "0", "pi")
    assert_no_error(res, "definite_integral")
    assert_equal(res["result"], "2")


# ── 6. compute_limit ────────────────────────────────────────────────────────

def test_compute_limit_sinx_over_x():
    """lim x→0 sin(x)/x = 1"""
    res = compute_limit("sin(x)/x", "x", "0")
    assert_no_error(res, "compute_limit")
    assert_equal(res["result"], "1")


def test_compute_limit_one_sided():
    """lim x→0⁺ 1/x = oo"""
    res = compute_limit("1/x", "x", "0", "+")
    assert_no_error(res, "compute_limit")
    assert_equal(res["result"], "oo")


# ── 7. partial_fractions ────────────────────────────────────────────────────

def test_partial_fractions():
    """Decompose 1/(x**2 - 1) into partial fractions"""
    res = partial_fractions("1/(x**2 - 1)")
    assert_no_error(res, "partial_fractions")
    result = res["result"]
    # Should contain two fraction terms with (x - 1) and (x + 1)
    assert "x - 1" in result and "x + 1" in result, f"Unexpected decomposition: {result}"


# ── 8. simplify_expression ──────────────────────────────────────────────────

def test_simplify_trig_identity():
    """sin(x)**2 + cos(x)**2 simplifies to 1"""
    res = simplify_expression("sin(x)**2 + cos(x)**2")
    assert_no_error(res, "simplify_expression")
    assert_equal(res["result"], "1")


def test_simplify_algebraic():
    """(x**2 - 1)/(x - 1) simplifies to x + 1"""
    res = simplify_expression("(x**2 - 1)/(x - 1)")
    assert_no_error(res, "simplify_expression")
    assert_equal(res["result"], "x + 1")


# ── 9. taylor_series ────────────────────────────────────────────────────────

def test_taylor_series_exp():
    """Taylor series of exp(x) around 0 with order 4"""
    res = taylor_series("exp(x)", "x", "0", 4)
    assert_no_error(res, "taylor_series")
    # Should contain at least the polynomial and terms keys
    assert "polynomial" in res, "Missing 'polynomial' in taylor_series result"
    assert "terms" in res, "Missing 'terms' in taylor_series result"
    # Polynomial should contain x**3/6, x**2/2, x, 1
    poly = res["polynomial"]
    assert "x" in poly, f"Unexpected polynomial: {poly}"


def test_taylor_series_sin():
    """Taylor series of sin(x) around 0 with order 6"""
    res = taylor_series("sin(x)", "x", "0", 6)
    assert_no_error(res, "taylor_series")
    assert "result" in res
    assert "x" in res["result"]


# ── 10. solve_system ────────────────────────────────────────────────────────

def test_solve_system_linear():
    """System: x + y = 5, x - y = 1 => x=3, y=2"""
    res = solve_system(["x + y = 5", "x - y = 1"])
    assert_no_error(res, "solve_system")
    sol = res["solutions"]
    assert "3" in sol and "2" in sol, f"Expected x=3, y=2; got {sol}"


def test_solve_system_three_vars():
    """System: x+y+z=6, x-y=0, y+z=4 => x=2, y=2, z=2"""
    res = solve_system(["x + y + z = 6", "x - y = 0", "y + z = 4"])
    assert_no_error(res, "solve_system")
    sol = res["solutions"]
    assert "2" in sol, f"Expected all vars = 2; got {sol}"


# ── 11. mean ─────────────────────────────────────────────────────────────────

def test_mean():
    """Mean of [1, 2, 3, 4, 5] = 3.0"""
    res = mean([1, 2, 3, 4, 5])
    assert_no_error(res, "mean")
    assert_close(res["result"], 3.0)


# ── 12. variance ─────────────────────────────────────────────────────────────

def test_variance():
    """Variance of [1, 2, 3, 4] = 1.25 (population variance)"""
    res = variance([1, 2, 3, 4])
    assert_no_error(res, "variance")
    assert_close(res["result"], 1.25)


# ── 13. standard_deviation ───────────────────────────────────────────────────

def test_standard_deviation():
    """Std dev of [2, 4, 4, 4, 5, 5, 7, 9] ≈ 2.0"""
    res = standard_deviation([2, 4, 4, 4, 5, 5, 7, 9])
    assert_no_error(res, "standard_deviation")
    assert_close(res["result"], 2.0)


# ── 14. median ───────────────────────────────────────────────────────────────

def test_median_odd():
    """Median of [1, 3, 5, 7, 9] = 5.0"""
    res = median([1, 3, 5, 7, 9])
    assert_no_error(res, "median")
    assert_close(res["result"], 5.0)


def test_median_even():
    """Median of [1, 2, 3, 4] = 2.5"""
    res = median([1, 2, 3, 4])
    assert_no_error(res, "median")
    assert_close(res["result"], 2.5)


# ── 15. mode ─────────────────────────────────────────────────────────────────

def test_mode():
    """Mode of [1, 2, 2, 3, 3, 3, 4] = 3.0"""
    res = mode([1, 2, 2, 3, 3, 3, 4])
    assert_no_error(res, "mode")
    assert_close(res["result"], 3.0)


# ── 16. correlation_coefficient ──────────────────────────────────────────────

def test_correlation_perfect_positive():
    """Perfect positive correlation between [1,2,3] and [2,4,6] = 1.0"""
    res = correlation_coefficient([1, 2, 3], [2, 4, 6])
    assert_no_error(res, "correlation_coefficient")
    assert_close(res["result"], 1.0)


def test_correlation_perfect_negative():
    """Perfect negative correlation between [1,2,3] and [6,4,2] = -1.0"""
    res = correlation_coefficient([1, 2, 3], [6, 4, 2])
    assert_no_error(res, "correlation_coefficient")
    assert_close(res["result"], -1.0)


# ── 17. linear_regression ───────────────────────────────────────────────────

def test_linear_regression_perfect_line():
    """Perfect line y = 2x + 1: slope=2.0, intercept=1.0"""
    res = linear_regression([(0, 1), (1, 3), (2, 5), (3, 7)])
    assert_no_error(res, "linear_regression")
    assert_close(res["slope"], 2.0)
    assert_close(res["intercept"], 1.0)


def test_linear_regression_approximate():
    """Approximate fit: slope ≈ 1.5"""
    res = linear_regression([(1, 2), (2, 3), (3, 5)])
    assert_no_error(res, "linear_regression")
    assert_close(res["slope"], 1.5)


# ── 18. matrix_addition ─────────────────────────────────────────────────────

def test_matrix_addition():
    """[[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]"""
    res = matrix_addition([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert_no_error(res, "matrix_addition")
    assert_equal(res["result"], [[6, 8], [10, 12]])


# ── 19. matrix_multiplication ───────────────────────────────────────────────

def test_matrix_multiplication():
    """[[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]]"""
    res = matrix_multiplication([[1, 2], [3, 4]], [[5, 6], [7, 8]])
    assert_no_error(res, "matrix_multiplication")
    assert_equal(res["result"], [[19, 22], [43, 50]])


# ── 20. matrix_transpose ────────────────────────────────────────────────────

def test_matrix_transpose():
    """Transpose of [[1,2,3],[4,5,6]] = [[1,4],[2,5],[3,6]]"""
    res = matrix_transpose([[1, 2, 3], [4, 5, 6]])
    assert_no_error(res, "matrix_transpose")
    assert_equal(res["result"], [[1, 4], [2, 5], [3, 6]])


# ── 21. matrix_determinant ──────────────────────────────────────────────────

def test_matrix_determinant_2x2():
    """det([[1,2],[3,4]]) = -2.0"""
    res = matrix_determinant([[1, 2], [3, 4]])
    assert_no_error(res, "matrix_determinant")
    assert_close(res["result"], -2.0)


def test_matrix_determinant_3x3():
    """det([[1,0,0],[0,2,0],[0,0,3]]) = 6.0 (diagonal matrix)"""
    res = matrix_determinant([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    assert_no_error(res, "matrix_determinant")
    assert_close(res["result"], 6.0)


# ── 22. matrix_eigenvalues ──────────────────────────────────────────────────

def test_matrix_eigenvalues_diagonal():
    """Eigenvalues of diagonal [[2,0],[0,3]] are [2.0, 3.0]"""
    res = matrix_eigenvalues([[2, 0], [0, 3]])
    assert_no_error(res, "matrix_eigenvalues")
    eigenvals = sorted(res["result"])
    assert_close(eigenvals[0], 2.0)
    assert_close(eigenvals[1], 3.0)


def test_matrix_eigenvalues_symmetric():
    """Eigenvalues of [[4,1],[1,4]] are [3.0, 5.0]"""
    res = matrix_eigenvalues([[4, 1], [1, 4]])
    assert_no_error(res, "matrix_eigenvalues")
    eigenvals = sorted(res["result"])
    assert_close(eigenvals[0], 3.0)
    assert_close(eigenvals[1], 5.0)


# ── 23. vector_dot_product ──────────────────────────────────────────────────

def test_vector_dot_product():
    """[1,2,3] · [4,5,6] = 32"""
    res = vector_dot_product((1, 2, 3), (4, 5, 6))
    assert_no_error(res, "vector_dot_product")
    assert_equal(res["result"], 32)


# ── 24. vector_cross_product ────────────────────────────────────────────────

def test_vector_cross_product():
    """[1,0,0] × [0,1,0] = [0,0,1]"""
    res = vector_cross_product((1, 0, 0), (0, 1, 0))
    assert_no_error(res, "vector_cross_product")
    assert_equal(res["result"], [0, 0, 1])


def test_vector_cross_product_general():
    """[1,2,3] × [4,5,6] = [-3,6,-3]"""
    res = vector_cross_product((1, 2, 3), (4, 5, 6))
    assert_no_error(res, "vector_cross_product")
    assert_equal(res["result"], [-3, 6, -3])


# ── 25. vector_magnitude ────────────────────────────────────────────────────

def test_vector_magnitude():
    """|[3, 4]| = 5.0"""
    res = vector_magnitude((3, 4))
    assert_no_error(res, "vector_magnitude")
    assert_close(res["result"], 5.0)


def test_vector_magnitude_3d():
    """|[1, 2, 2]| = 3.0"""
    res = vector_magnitude((1, 2, 2))
    assert_no_error(res, "vector_magnitude")
    assert_close(res["result"], 3.0)


# ── 26. summation ────────────────────────────────────────────────────────────

def test_summation_squares():
    """Σ(x², x=0..10) = 385"""
    res = summation("x**2", 0, 10)
    assert_no_error(res, "summation")
    assert_equal(res["result"], 385)


def test_summation_linear():
    """Σ(x, x=1..100) = 5050"""
    res = summation("x", 1, 100)
    assert_no_error(res, "summation")
    assert_equal(res["result"], 5050)


# ── 27. expand ───────────────────────────────────────────────────────────────

def test_expand_binomial_square():
    """(x + 1)**2 = x**2 + 2*x + 1"""
    res = expand("(x + 1)**2")
    assert_no_error(res, "expand")
    assert_equal(res["result"], "x**2 + 2*x + 1")


def test_expand_product():
    """(x + 1)*(x - 1) = x**2 - 1"""
    res = expand("(x + 1)*(x - 1)")
    assert_no_error(res, "expand")
    assert_equal(res["result"], "x**2 - 1")


# ── 28. factorize ───────────────────────────────────────────────────────────

def test_factorize_perfect_square():
    """x**2 + 2*x + 1 = (x + 1)**2"""
    res = factorize("x**2 + 2*x + 1")
    assert_no_error(res, "factorize")
    assert_equal(res["result"], "(x + 1)**2")


def test_factorize_difference_of_squares():
    """x**2 - 1 = (x - 1)*(x + 1)"""
    res = factorize("x**2 - 1")
    assert_no_error(res, "factorize")
    assert_equal(res["result"], "(x - 1)*(x + 1)")


# ── Runner ───────────────────────────────────────────────────────────────────

ALL_TESTS = [
    # 1. calculate (basic expression evaluator)
    ("calculate -- basic arithmetic", test_calculate_basic_arithmetic),
    ("calculate -- trig function", test_calculate_trig),
    ("calculate -- sqrt", test_calculate_sqrt),
    ("calculate -- power", test_calculate_power),
    ("calculate -- log", test_calculate_log),
    # 2. solve_equation (algebraic solver)
    ("solve_equation -- linear", test_solve_equation_linear),
    ("solve_equation -- quadratic", test_solve_equation_quadratic),
    # 3. differentiate (symbolic derivative)
    ("differentiate -- polynomial", test_differentiate_polynomial),
    ("differentiate -- trig", test_differentiate_trig),
    # 4. integrate (indefinite integral)
    ("integrate -- polynomial", test_integrate_polynomial),
    ("integrate -- trig", test_integrate_trig),
    # 5. definite_integral (bounded integral)
    ("definite_integral -- polynomial", test_definite_integral_polynomial),
    ("definite_integral -- trig", test_definite_integral_trig),
    # 6. compute_limit
    ("compute_limit -- sin(x)/x", test_compute_limit_sinx_over_x),
    ("compute_limit -- one-sided", test_compute_limit_one_sided),
    # 7. partial_fractions
    ("partial_fractions -- rational", test_partial_fractions),
    # 8. simplify_expression
    ("simplify_expression -- trig identity", test_simplify_trig_identity),
    ("simplify_expression -- algebraic", test_simplify_algebraic),
    # 9. taylor_series
    ("taylor_series -- exp(x)", test_taylor_series_exp),
    ("taylor_series -- sin(x)", test_taylor_series_sin),
    # 10. solve_system (system of equations)
    ("solve_system -- 2 vars", test_solve_system_linear),
    ("solve_system -- 3 vars", test_solve_system_three_vars),
    # 11. mean
    ("mean", test_mean),
    # 12. variance
    ("variance", test_variance),
    # 13. standard_deviation
    ("standard_deviation", test_standard_deviation),
    # 14. median
    ("median -- odd count", test_median_odd),
    ("median -- even count", test_median_even),
    # 15. mode
    ("mode", test_mode),
    # 16. correlation_coefficient
    ("correlation_coefficient -- perfect positive", test_correlation_perfect_positive),
    ("correlation_coefficient -- perfect negative", test_correlation_perfect_negative),
    # 17. linear_regression
    ("linear_regression -- perfect line", test_linear_regression_perfect_line),
    ("linear_regression -- approximate", test_linear_regression_approximate),
    # 18. matrix_addition
    ("matrix_addition", test_matrix_addition),
    # 19. matrix_multiplication
    ("matrix_multiplication", test_matrix_multiplication),
    # 20. matrix_transpose
    ("matrix_transpose", test_matrix_transpose),
    # 21. matrix_determinant
    ("matrix_determinant -- 2x2", test_matrix_determinant_2x2),
    ("matrix_determinant -- 3x3", test_matrix_determinant_3x3),
    # 22. matrix_eigenvalues
    ("matrix_eigenvalues -- diagonal", test_matrix_eigenvalues_diagonal),
    ("matrix_eigenvalues -- symmetric", test_matrix_eigenvalues_symmetric),
    # 23. vector_dot_product
    ("vector_dot_product", test_vector_dot_product),
    # 24. vector_cross_product
    ("vector_cross_product -- unit vectors", test_vector_cross_product),
    ("vector_cross_product -- general", test_vector_cross_product_general),
    # 25. vector_magnitude
    ("vector_magnitude -- 2D", test_vector_magnitude),
    ("vector_magnitude -- 3D", test_vector_magnitude_3d),
    # 26. summation
    ("summation -- squares", test_summation_squares),
    ("summation -- linear (Gauss)", test_summation_linear),
    # 27. expand
    ("expand -- binomial square", test_expand_binomial_square),
    ("expand -- product", test_expand_product),
    # 28. factorize
    ("factorize -- perfect square", test_factorize_perfect_square),
    ("factorize -- difference of squares", test_factorize_difference_of_squares),
]


def main():
    print("=" * 70)
    print("  Calculator MCP Server -- Tool Test Suite (28 tools)")
    print("=" * 70)
    print()

    # Group tests by tool number for cleaner output
    current_tool = None
    for name, func in ALL_TESTS:
        tool = name.split(" -- ")[0].split(" ")[0]
        if tool != current_tool:
            current_tool = tool
            print(f"\n-- {tool} --")
        run_test(name, func)

    # Summary
    total = PASSED + FAILED
    print()
    print("=" * 70)
    print(f"  RESULTS: {PASSED}/{total} passed, {FAILED} failed")
    print("=" * 70)

    if ERRORS:
        print("\nFailed tests:")
        for name, exc in ERRORS:
            print(f"  - {name}: {exc}")

    # Non-zero exit code if any test failed
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
