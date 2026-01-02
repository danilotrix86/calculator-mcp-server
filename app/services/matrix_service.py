"""
Advanced matrix operations service
Provides comprehensive matrix calculations and analysis
"""
import numpy as np
from scipy import linalg
from typing import List, Dict, Any, Tuple, Optional
import logging


def perform_matrix_operation(
    operation: str,
    matrix_a: List[List[float]],
    matrix_b: Optional[List[List[float]]] = None,
    scalar: Optional[float] = None,
    power: Optional[int] = None
) -> Dict[str, Any]:
    """
    Perform various matrix operations
    
    Args:
        operation: The operation to perform
        matrix_a: First matrix
        matrix_b: Second matrix (for binary operations)
        scalar: Scalar value (for scalar operations)
        power: Power value (for matrix power)
        
    Returns:
        Dictionary with result, result_type, and optional explanation
    """
    try:
        mat_a = np.array(matrix_a, dtype=float)
        
        # Validate matrix
        if mat_a.ndim != 2:
            return {"error": "Matrix A must be 2-dimensional", "result": None, "result_type": "error"}
        
        if operation == "add":
            if matrix_b is None:
                return {"error": "La matrice B è richiesta per l'addizione", "result": None, "result_type": "error"}
            mat_b = np.array(matrix_b, dtype=float)
            result = np.add(mat_a, mat_b)
            return {
                "result": result.tolist(),
                "result_type": "matrix",
                "explanation": f"Sommate due matrici di dimensione {mat_a.shape}. Risultato: {result.shape}"
            }
        
        elif operation == "subtract":
            if matrix_b is None:
                return {"error": "La matrice B è richiesta per la sottrazione", "result": None, "result_type": "error"}
            mat_b = np.array(matrix_b, dtype=float)
            result = np.subtract(mat_a, mat_b)
            return {
                "result": result.tolist(),
                "result_type": "matrix",
                "explanation": f"Sottratta matrice B da matrice A. Dimensione: {result.shape}"
            }
        
        elif operation == "multiply":
            if matrix_b is None:
                return {"error": "La matrice B è richiesta per la moltiplicazione", "result": None, "result_type": "error"}
            mat_b = np.array(matrix_b, dtype=float)
            result = np.dot(mat_a, mat_b)
            return {
                "result": result.tolist(),
                "result_type": "matrix",
                "explanation": f"Moltiplicata matrice A ({mat_a.shape[0]}×{mat_a.shape[1]}) per matrice B ({mat_b.shape[0]}×{mat_b.shape[1]}). Risultato: {result.shape[0]}×{result.shape[1]}"
            }
        
        elif operation == "scalar_multiply":
            if scalar is None:
                return {"error": "Il valore scalare è richiesto", "result": None, "result_type": "error"}
            result = mat_a * scalar
            return {
                "result": result.tolist(),
                "result_type": "matrix",
                "explanation": f"Moltiplicata matrice per scalare {scalar}"
            }
        
        elif operation == "transpose":
            result = np.transpose(mat_a)
            return {
                "result": result.tolist(),
                "result_type": "matrix",
                "explanation": f"Trasposta matrice da {mat_a.shape} a {result.shape}"
            }
        
        elif operation == "determinant":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "Il determinante richiede una matrice quadrata", "result": None, "result_type": "error"}
            result = float(np.linalg.det(mat_a))
            return {
                "result": round(result, 10),
                "result_type": "scalar",
                "explanation": f"Calcolato determinante della matrice {mat_a.shape[0]}×{mat_a.shape[1]}"
            }
        
        elif operation == "inverse":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "L'inversa richiede una matrice quadrata", "result": None, "result_type": "error"}
            try:
                result = np.linalg.inv(mat_a)
                return {
                    "result": result.tolist(),
                    "result_type": "matrix",
                    "explanation": f"Calcolata inversa della matrice {mat_a.shape[0]}×{mat_a.shape[1]}"
                }
            except np.linalg.LinAlgError:
                return {"error": "La matrice è singolare (non invertibile)", "result": None, "result_type": "error"}
        
        elif operation == "eigenvalues":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "Gli autovalori richiedono una matrice quadrata", "result": None, "result_type": "error"}
            eigenvalues = np.linalg.eigvals(mat_a)
            # Handle complex eigenvalues
            if np.iscomplexobj(eigenvalues):
                result = [{"real": float(ev.real), "imag": float(ev.imag)} for ev in eigenvalues]
            else:
                result = [float(ev) for ev in eigenvalues]
            return {
                "result": result,
                "result_type": "vector",
                "explanation": f"Calcolati {len(eigenvalues)} autovalori"
            }
        
        elif operation == "eigenvectors":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "Gli autovettori richiedono una matrice quadrata", "result": None, "result_type": "error"}
            eigenvalues, eigenvectors = np.linalg.eig(mat_a)
            
            # Format eigenvalues
            if np.iscomplexobj(eigenvalues):
                evals = [{"real": float(ev.real), "imag": float(ev.imag)} for ev in eigenvalues]
            else:
                evals = [float(ev) for ev in eigenvalues]
            
            # Format eigenvectors
            if np.iscomplexobj(eigenvectors):
                evecs = [[{"real": float(v.real), "imag": float(v.imag)} for v in vec] for vec in eigenvectors.T]
            else:
                evecs = eigenvectors.T.tolist()
            
            return {
                "result": {
                    "eigenvalues": evals,
                    "eigenvectors": evecs
                },
                "result_type": "decomposition",
                "explanation": f"Calcolati autovalori e autovettori per matrice {mat_a.shape[0]}×{mat_a.shape[1]}"
            }
        
        elif operation == "rank":
            result = int(np.linalg.matrix_rank(mat_a))
            return {
                "result": result,
                "result_type": "scalar",
                "explanation": f"Rango della matrice: {result} (numero massimo di righe/colonne linearmente indipendenti)"
            }
        
        elif operation == "trace":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "La traccia richiede una matrice quadrata", "result": None, "result_type": "error"}
            result = float(np.trace(mat_a))
            return {
                "result": round(result, 10),
                "result_type": "scalar",
                "explanation": f"Traccia (somma degli elementi diagonali): {result}"
            }
        
        elif operation == "lu_decomposition":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "La decomposizione LU richiede una matrice quadrata", "result": None, "result_type": "error"}
            P, L, U = linalg.lu(mat_a)
            return {
                "result": {
                    "P": P.tolist(),
                    "L": L.tolist(),
                    "U": U.tolist()
                },
                "result_type": "decomposition",
                "explanation": "Decomposizione LU: A = PLU dove P è permutazione, L è triangolare inferiore, U è triangolare superiore"
            }
        
        elif operation == "qr_decomposition":
            Q, R = np.linalg.qr(mat_a)
            return {
                "result": {
                    "Q": Q.tolist(),
                    "R": R.tolist()
                },
                "result_type": "decomposition",
                "explanation": "Decomposizione QR: A = QR dove Q è ortogonale e R è triangolare superiore"
            }
        
        elif operation == "svd":
            U, s, Vt = np.linalg.svd(mat_a)
            return {
                "result": {
                    "U": U.tolist(),
                    "S": s.tolist(),
                    "Vt": Vt.tolist()
                },
                "result_type": "decomposition",
                "explanation": "Decomposizione ai Valori Singolari: A = U·Σ·V^T"
            }
        
        elif operation == "power":
            if mat_a.shape[0] != mat_a.shape[1]:
                return {"error": "La potenza matriciale richiede una matrice quadrata", "result": None, "result_type": "error"}
            if power is None:
                return {"error": "Il valore della potenza è richiesto", "result": None, "result_type": "error"}
            result = np.linalg.matrix_power(mat_a, power)
            return {
                "result": result.tolist(),
                "result_type": "matrix",
                "explanation": f"Elevata matrice alla potenza {power}"
            }
        
        elif operation == "rref":
            # Reduced Row Echelon Form using sympy for exact computation
            from sympy import Matrix
            sympy_mat = Matrix(matrix_a)
            rref_mat, pivot_cols = sympy_mat.rref()
            result = [[float(val) for val in row] for row in rref_mat.tolist()]
            return {
                "result": result,
                "result_type": "matrix",
                "explanation": f"Forma Ridotta a Scala per Righe. Colonne pivot: {list(pivot_cols)}",
                "properties": {"pivot_columns": list(pivot_cols)}
            }
        
        elif operation == "norm":
            result = float(np.linalg.norm(mat_a))
            return {
                "result": round(result, 10),
                "result_type": "scalar",
                "explanation": f"Norma di Frobenius (radice quadrata della somma dei quadrati degli elementi): {result}"
            }
        
        else:
            return {"error": f"Operazione sconosciuta: {operation}", "result": None, "result_type": "error"}
            
    except Exception as e:
        logging.error(f"Matrix operation error: {str(e)}")
        return {"error": str(e), "result": None, "result_type": "error"}


def get_matrix_properties(matrix: List[List[float]]) -> Dict[str, Any]:
    """
    Get comprehensive properties of a matrix
    
    Args:
        matrix: Input matrix as 2D list
        
    Returns:
        Dictionary with matrix properties
    """
    try:
        mat = np.array(matrix, dtype=float)
        
        if mat.ndim != 2:
            return {"error": "Matrix must be 2-dimensional"}
        
        rows, cols = mat.shape
        is_square = rows == cols
        
        properties = {
            "rows": rows,
            "cols": cols,
            "is_square": is_square,
            "is_symmetric": False,
            "is_diagonal": False,
            "is_identity": False,
            "is_invertible": None,
            "determinant": None,
            "rank": None,
            "trace": None,
            "condition_number": None
        }
        
        # Check if symmetric
        if is_square:
            properties["is_symmetric"] = bool(np.allclose(mat, mat.T))
            properties["is_diagonal"] = bool(np.allclose(mat, np.diag(np.diag(mat))))
            properties["is_identity"] = bool(np.allclose(mat, np.eye(rows)))
            
            # Calculate determinant
            try:
                det = float(np.linalg.det(mat))
                properties["determinant"] = round(det, 10)
                properties["is_invertible"] = abs(det) > 1e-10
            except:
                pass
            
            # Calculate trace
            try:
                properties["trace"] = float(np.trace(mat))
            except:
                pass
            
            # Calculate condition number
            try:
                properties["condition_number"] = float(np.linalg.cond(mat))
            except:
                pass
        
        # Calculate rank
        try:
            properties["rank"] = int(np.linalg.matrix_rank(mat))
        except:
            pass
        
        return properties
        
    except Exception as e:
        logging.error(f"Error calculating matrix properties: {str(e)}")
        return {"error": str(e)}

