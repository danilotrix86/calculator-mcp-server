from fastapi import APIRouter, HTTPException, Request
import logging

from app.schemas.matrix import (
    MatrixOperationRequest,
    MatrixOperationResponse,
    MatrixPropertiesRequest,
    MatrixPropertiesResponse
)
from app.services.matrix_service import perform_matrix_operation, get_matrix_properties
from app.middleware.rate_limit import limiter
from app.config import rate_limit_config

router = APIRouter(tags=["matrix"])


@router.post("/matrix/operation", response_model=MatrixOperationResponse)
@limiter.limit(rate_limit_config.MATRIX)
async def matrix_operation_endpoint(request: Request, payload: MatrixOperationRequest) -> MatrixOperationResponse:
    """
    Perform a matrix operation
    
    Supported operations:
    - add: Add two matrices
    - subtract: Subtract matrix B from matrix A
    - multiply: Multiply two matrices
    - scalar_multiply: Multiply matrix by a scalar
    - transpose: Transpose a matrix
    - determinant: Calculate determinant
    - inverse: Calculate inverse
    - eigenvalues: Calculate eigenvalues
    - eigenvectors: Calculate eigenvalues and eigenvectors
    - rank: Calculate matrix rank
    - trace: Calculate trace
    - lu_decomposition: LU decomposition
    - qr_decomposition: QR decomposition
    - svd: Singular Value Decomposition
    - power: Raise matrix to a power
    - rref: Reduced Row Echelon Form
    - norm: Calculate Frobenius norm
    """
    result = perform_matrix_operation(
        operation=payload.operation,
        matrix_a=payload.matrix_a,
        matrix_b=payload.matrix_b,
        scalar=payload.scalar,
        power=payload.power
    )
    
    if result.get("error"):
        logging.error(f"Matrix operation error: {result['error']}")
        return MatrixOperationResponse(
            result=None,
            result_type="error",
            error=result["error"]
        )
    
    return MatrixOperationResponse(
        result=result["result"],
        result_type=result["result_type"],
        explanation=result.get("explanation"),
        properties=result.get("properties")
    )


@router.post("/matrix/properties", response_model=MatrixPropertiesResponse)
@limiter.limit(rate_limit_config.MATRIX)
async def matrix_properties_endpoint(request: Request, payload: MatrixPropertiesRequest) -> MatrixPropertiesResponse:
    """
    Get comprehensive properties of a matrix
    """
    properties = get_matrix_properties(payload.matrix)
    
    if "error" in properties:
        logging.error(f"Matrix properties error: {properties['error']}")
        return MatrixPropertiesResponse(
            rows=0,
            cols=0,
            is_square=False,
            is_symmetric=False,
            is_diagonal=False,
            is_identity=False,
            error=properties["error"]
        )
    
    return MatrixPropertiesResponse(**properties)



