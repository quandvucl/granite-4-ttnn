"""Test native ttnn.rms_norm implementation."""
import ttnn
import torch
from tt_ops.normalization import TTRMSNorm
from tt_ops.base import to_tt_tensor, to_torch_tensor
from tt_ops.fused_ops import NATIVE_RMS_NORM_AVAILABLE

print(f"Native ttnn.rms_norm available: {NATIVE_RMS_NORM_AVAILABLE}")

device = ttnn.open_device(device_id=0)
try:
    batch_size, seq_len, hidden_size = 2, 4, 16
    eps = 1e-5

    # Create test data
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    weight = torch.ones(hidden_size, dtype=torch.bfloat16)

    # CPU reference implementation (HF-compatible with FP32 intermediate)
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normalized = x * torch.rsqrt(variance + eps)
    cpu_output = x_normalized * weight

    # TT implementation with native rms_norm
    print("\nTesting TTRMSNorm with native ttnn.rms_norm...")
    tt_norm = TTRMSNorm(device, weight, eps=eps)
    x_tt = to_tt_tensor(x, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    tt_output_tt = tt_norm(x_tt)
    tt_output = to_torch_tensor(tt_output_tt, target_shape=(batch_size, seq_len, hidden_size))

    # Compare
    diff = torch.abs(cpu_output - tt_output)
    max_diff = diff.max().item()
    rel_error = (diff / (torch.abs(cpu_output) + 1e-8)).mean().item()

    print(f"  Max absolute difference: {max_diff:.6f}")
    print(f"  Mean relative error: {rel_error:.6f}")
    print(f"  {'✓ PASSED' if max_diff < 0.02 else '✗ FAILED'} (BF16 tolerance: 0.02)")

    # Test fused residual + rmsnorm
    print("\nTesting fused residual add + RMSNorm...")
    residual = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)
    scale = 0.22

    # Reference
    hidden_ref = residual + x * scale
    variance_ref = hidden_ref.pow(2).mean(-1, keepdim=True)
    norm_ref = hidden_ref * torch.rsqrt(variance_ref + eps) * weight

    # TT fused
    residual_tt = to_tt_tensor(residual, device, layout=ttnn.ROW_MAJOR_LAYOUT)
    x_tt = to_tt_tensor(x, device, layout=ttnn.ROW_MAJOR_LAYOUT)

    hidden_tt, norm_tt = tt_norm.forward_fused(x_tt, residual_tt, scale)

    hidden_out = to_torch_tensor(hidden_tt, target_shape=(batch_size, seq_len, hidden_size))
    norm_out = to_torch_tensor(norm_tt, target_shape=(batch_size, seq_len, hidden_size))

    # Compare
    diff_norm = torch.abs(norm_ref - norm_out)
    max_diff_norm = diff_norm.max().item()

    print(f"  Max difference (normalized): {max_diff_norm:.6f}")
    print(f"  {'✓ PASSED' if max_diff_norm < 0.02 else '✗ FAILED'} (BF16 tolerance: 0.02)")

finally:
    ttnn.close_device(device)

print("\n✓ All tests completed!")
