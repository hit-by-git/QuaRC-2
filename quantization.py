"""
Quantization utilities - LSQ+ and fake quantization operations
Based on: "LSQ+: Improving low-bit quantization through learnable offsets and better initialization"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import SYMMETRIC_QUANTIZATION


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
        self.register_buffer('zero_point', torch.zeros(1))
    
    def forward(self, x):
        """
        Fake quantization forward pass
        x_q = s * clamp(round(x_r / s), -Q_N, Q_P)
        """
        # Clamp scale to avoid division by zero
        scale = torch.clamp(self.scale, min=1e-8)
        
        # Quantize
        x_normalized = x / scale
        x_rounded = torch.round(x_normalized)
        x_clamped = torch.clamp(x_rounded, self.qn, self.qp)
        x_quantized = x_clamped * scale
        
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
            # Forward: fake quantization
            x_quant = self.fake_quant(x)
            # Backward: straight through (gradients pass unchanged)
            return x_quant + (x - x_quant).detach()
        else:
            return self.fake_quant(x)
