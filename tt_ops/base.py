import ttnn
import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union


class TTOperation(ABC):
    """Base class for all TTNN operations."""

    def __init__(self, device, dtype=ttnn.bfloat16):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass of the operation."""
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def to_tt_tensor(
    torch_tensor: torch.Tensor,
    device,
    dtype=ttnn.bfloat16,
    layout=ttnn.ROW_MAJOR_LAYOUT
) -> ttnn.Tensor:
    """
    Convert PyTorch tensor to TTNN tensor.

    Args:
        torch_tensor: Input PyTorch tensor
        device: TTNN device
        dtype: Target dtype (default: bfloat16)
        layout: Target layout (default: ROW_MAJOR_LAYOUT for transfers)

    Returns:
        TTNN tensor

    Note:
        Always use ROW_MAJOR_LAYOUT for CPU→TT transfers to prevent
        tile padding corruption. Convert to TILE_LAYOUT for computation.
    """
    if torch_tensor is None:
        return None

    # Ensure tensor is contiguous
    if not torch_tensor.is_contiguous():
        torch_tensor = torch_tensor.contiguous()

    # Use ttnn.Tensor constructor (overload #8) which accepts torch tensors
    tt_tensor = ttnn.Tensor(
        torch_tensor,
        data_type=dtype,
        device=device,
        layout=layout
    )

    return tt_tensor


def to_torch_tensor(
    tt_tensor: ttnn.Tensor,
    target_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """
    Convert TTNN tensor back to PyTorch tensor.

    Args:
        tt_tensor: Input TTNN tensor
        target_shape: Optional target shape to reshape to

    Returns:
        PyTorch tensor

    Note:
        TILE_LAYOUT may add padding, so always force reshape to
        expected shape after conversion.
    """
    if tt_tensor is None:
        return None

    # Use the tensor's to_torch() method
    torch_tensor = tt_tensor.to_torch()

    # Force reshape to expected shape to undo padding
    if target_shape is not None:
        torch_tensor = torch_tensor.view(*target_shape)

    return torch_tensor


def to_tile_layout(tt_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    Convert TTNN tensor to TILE layout for optimal computation.

    Args:
        tt_tensor: Input TTNN tensor in ROW_MAJOR layout

    Returns:
        TTNN tensor in TILE layout
    """
    if tt_tensor is None:
        return None

    return tt_tensor.to(ttnn.TILE_LAYOUT)


def to_row_major_layout(tt_tensor: ttnn.Tensor) -> ttnn.Tensor:
    """
    Convert TTNN tensor to ROW_MAJOR layout.

    Args:
        tt_tensor: Input TTNN tensor in TILE layout

    Returns:
        TTNN tensor in ROW_MAJOR layout
    """
    if tt_tensor is None:
        return None

    return tt_tensor.to(ttnn.ROW_MAJOR_LAYOUT)


def validate_shape(
    tensor: Union[torch.Tensor, ttnn.Tensor],
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> None:
    """
    Validate tensor shape matches expected shape.

    Args:
        tensor: Input tensor
        expected_shape: Expected shape tuple
        name: Tensor name for error messages

    Raises:
        ValueError: If shape doesn't match
    """
    if isinstance(tensor, ttnn.Tensor):
        actual_shape = tensor.shape
    else:
        actual_shape = tensor.shape

    if len(actual_shape) != len(expected_shape):
        raise ValueError(
            f"{name} rank mismatch: expected {len(expected_shape)}, "
            f"got {len(actual_shape)} (shape: {actual_shape})"
        )

    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise ValueError(
                f"{name} shape mismatch at dim {i}: "
                f"expected {expected}, got {actual} (full shape: {actual_shape})"
            )


def tt_linear(
    input_tt: ttnn.Tensor,
    weight_tt: ttnn.Tensor,
    bias_tt: Optional[ttnn.Tensor] = None
) -> ttnn.Tensor:
    """
    Perform linear transformation on TTNN tensors.

    Args:
        input_tt: Input tensor [batch, seq, in_features] (ROW_MAJOR or TILE)
        weight_tt: Weight tensor [in_features, out_features] (TILE)
        bias_tt: Optional bias tensor [out_features]

    Returns:
        Output tensor [batch, seq, out_features] (TILE)
    """
    # Ensure input is in TILE layout for optimal matmul
    if input_tt.layout != ttnn.TILE_LAYOUT:
        input_tt = to_tile_layout(input_tt)

    # Weight should already be in TILE from cache
    # Matmul: [batch, seq, in_features] @ [in_features, out_features]
    output = input_tt @ weight_tt

    if bias_tt is not None:
        output = output + bias_tt

    return output


def reshape_for_broadcast(
    tensor_tt: ttnn.Tensor,
    target_shape: Tuple[int, ...]
) -> ttnn.Tensor:
    """
    Reshape tensor for broadcasting operations.

    Args:
        tensor_tt: Input TTNN tensor
        target_shape: Target shape for broadcasting

    Returns:
        Reshaped TTNN tensor
    """
    return tensor_tt.reshape(target_shape)
