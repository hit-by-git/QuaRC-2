"""
Quantization utilities - LSQ+ and fake quantization operations
Based on: "LSQ+: Improving low-bit quantization through learnable offsets and better initialization"

# QuaRC Step 0: LSQ+ quantization
# - Learnable scale parameters
# - Fake quantization for weights/activations
# - Straight-through estimator for gradients
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import types
from config import SYMMETRIC_QUANTIZATION


def _gradient_scale(x, scale):
    """Scale gradients flowing through x while preserving forward value."""
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


class FakeQuantization(nn.Module):
    """Fake quantization node for simulating low-bit quantization during training"""
    
    def __init__(self, bits, symmetric=True):
        super().__init__()
        self.bits = bits
        self.symmetric = symmetric
        
        if symmetric:
            self.qn = -(2 ** (bits - 1))
            self.qp = 2 ** (bits - 1) - 1
        else:
            self.qn = 0
            self.qp = 2 ** bits - 1
        
        # Learnable scale parameter
        self.register_parameter('scale', nn.Parameter(torch.ones(1)))
        # LSQ+ style learnable offset (beta)
        self.register_parameter('offset', nn.Parameter(torch.zeros(1)))
    
    def forward(self, x):
        """
        Fake quantization forward pass
        x_q = s * clamp(round(x_r / s), -Q_N, Q_P)
        """
        # LSQ+ forward quantization step used by QuaRC
        # Clamp scale to avoid division by zero
        scale = torch.clamp(self.scale, min=1e-8)
        grad_scale = 1.0 / ((x.numel() * max(self.qp, 1)) ** 0.5)
        scale = _gradient_scale(scale, grad_scale)
        offset = self.offset
        
        # Quantize
        x_normalized = (x - offset) / scale
        x_rounded = torch.round(x_normalized)
        x_clamped = torch.clamp(x_rounded, self.qn, self.qp)
        x_quantized = x_clamped * scale + offset
        
        return x_quantized
    
    def initialize_scale(self, x, method='absmax'):
        """Initialize scale parameter"""
        if method == 'absmax':
            # Initialize scale based on absolute maximum value
            scale_init = x.abs().max() / self.qp
        else:
            # Initialize based on percentile
            scale_init = torch.quantile(x.abs(), 0.9999) / self.qp
        
        self.scale.data = torch.clamp(scale_init, min=1e-8)
        self.offset.data = x.mean().detach().reshape_as(self.offset)


class QuantizationModule(nn.Module):
    """Module to add fake quantization to model weights and activations"""
    
    def __init__(self, weight_bits, activation_bits, symmetric=True):
        super().__init__()
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        self.symmetric = symmetric
        
        # Create quantization modules
        self.weight_quantizer = FakeQuantization(weight_bits, symmetric)
        self.activation_quantizer = FakeQuantization(activation_bits, symmetric)
    
    def quantize_weight(self, weight):
        """Quantize weight tensor"""
        if self.weight_bits < 32:
            return self.weight_quantizer(weight)
        return weight
    
    def quantize_activation(self, activation):
        """Quantize activation tensor"""
        if self.activation_bits < 32:
            return self.activation_quantizer(activation)
        return activation


def init_quantization_params(model, train_loader, weight_bits, activation_bits):
    """
    Initialize quantization parameters using first batch of data
    """
    model.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(next(model.parameters()).device)
            
            # Forward pass to initialize scales
            outputs = model(inputs)
            break


def apply_fake_quantization_to_weights(model, weight_bits, symmetric=True):
    """
    Apply fake quantization to all weights in the model
    """
    quantizers = {}
    
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            quantizer = FakeQuantization(weight_bits, symmetric)
            quantizer.initialize_scale(param.data)
            quantizers[name] = quantizer
    
    return quantizers


def straight_through_estimator(x):
    """
    Straight-through estimator for gradient approximation in quantization
    Gradients flow through as if there was no quantization
    """
    return x


class StraightThroughQuantizer(nn.Module):
    """Quantization layer with straight-through estimator for gradients"""
    
    def __init__(self, bits, symmetric=True):
        super().__init__()
        self.fake_quant = FakeQuantization(bits, symmetric)
    
    def forward(self, x):
        """Forward with fake quantization, backward with straight-through"""
        if self.training:
            # Forward should use quantized values while backward approximates identity.
            x_quant = self.fake_quant(x)
            return x + (x_quant - x).detach()
        else:
            return self.fake_quant(x)


def _patch_quantized_forward(module, activation_bits):
    """Replace Conv/Linear forward with a weight-quantized forward."""
    module._original_forward = module.forward

    def _forward(self, x):
        quant_weight = self.weight_quantizer(self.weight)

        if isinstance(self, nn.Conv2d):
            out = F.conv2d(
                x,
                quant_weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        else:
            out = F.linear(x, quant_weight, self.bias)

        if activation_bits < 32 and hasattr(self, 'activation_quantizer'):
            out = self.activation_quantizer(out)

        return out

    module.forward = types.MethodType(_forward, module)


def attach_lsq_plus_quantization(model, weight_bits, activation_bits, symmetric=True):
    """Attach LSQ+ quantizers directly to Conv2d/Linear modules in the model."""
    if weight_bits >= 32 and activation_bits >= 32:
        return model

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if weight_bits < 32:
                module.weight_quantizer = StraightThroughQuantizer(weight_bits, symmetric)
                module.weight_quantizer.fake_quant.initialize_scale(module.weight.data)

            if activation_bits < 32:
                module.activation_quantizer = StraightThroughQuantizer(activation_bits, symmetric)

            if weight_bits < 32:
                _patch_quantized_forward(module, activation_bits)

    return model
