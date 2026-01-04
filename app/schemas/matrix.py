from pydantic import BaseModel, Field
from typing import List, Optional, Any


class MatrixOperationRequest(BaseModel):
    """Request schema for matrix operations"""
    operation: str = Field(..., description="The operation to perform: add, subtract, multiply, transpose, determinant, inverse, eigenvalues, eigenvectors, rank, trace, lu_decomposition, qr_decomposition, svd, power, rref")
    matrix_a: List[List[float]] = Field(..., description="First matrix as 2D array")
    matrix_b: Optional[List[List[float]]] = Field(None, description="Second matrix (for binary operations)")
    scalar: Optional[float] = Field(None, description="Scalar value (for scalar operations)")
    power: Optional[int] = Field(None, description="Power value (for matrix power)")


class MatrixOperationResponse(BaseModel):
    """Response schema for matrix operations"""
    result: Any = Field(..., description="The result of the operation")
    result_type: str = Field(..., description="Type of result: matrix, scalar, vector, decomposition")
    explanation: Optional[str] = Field(None, description="Step-by-step explanation")
    properties: Optional[dict] = Field(None, description="Matrix properties")
    error: Optional[str] = Field(None, description="Error message if operation failed")


class MatrixPropertiesRequest(BaseModel):
    """Request schema for getting matrix properties"""
    matrix: List[List[float]] = Field(..., description="Matrix as 2D array")


class MatrixPropertiesResponse(BaseModel):
    """Response schema for matrix properties"""
    rows: int
    cols: int
    is_square: bool
    determinant: Optional[float] = None
    rank: Optional[int] = None
    trace: Optional[float] = None
    is_symmetric: bool
    is_diagonal: bool
    is_identity: bool
    is_invertible: Optional[bool] = None
    condition_number: Optional[float] = None
    error: Optional[str] = None


