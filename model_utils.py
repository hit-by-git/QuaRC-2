"""
Model utilities and helpers
"""
import torch
import torch.nn as nn
from torchvision import models
import os


def get_model(model_name='mobilenetv2', num_classes=100, pretrained=True, device='cuda'):
    """
    Load a pre-trained model
    
    Args:
        model_name: Name of model ('mobilenetv2', 'resnet18', etc.)
        num_classes: Number of output classes
        pretrained: Whether to load pretrained weights
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    if model_name.lower() == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=pretrained)
        # Modify classifier for CIFAR-100
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    elif model_name.lower() == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    elif model_name.lower() == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    model = model.to(device)
    return model


def quantize_model(model, weight_bits, activation_bits):
    """
    Apply quantization to model
    
    For simplicity, we'll use fake quantization on weights and activations
    In full implementation, this would replace layers with quantized versions
    
    Args:
        model: Model to quantize
        weight_bits: Number of bits for weights
        activation_bits: Number of bits for activations
    
    Returns:
        Quantized model
    """
    # Deep copy the model
    import copy
    q_model = copy.deepcopy(model)
    
    # Store quantization parameters
    q_model.weight_bits = weight_bits
    q_model.activation_bits = activation_bits
    
    # For LSQ+, we'll apply learnable scale factors to weights
    for name, module in q_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            # Add learnable scale parameter
            if hasattr(module, 'weight'):
                # Initialize scale based on weight statistics
                weight = module.weight.data
                scale = weight.abs().max() / (2 ** (weight_bits - 1))
                if not hasattr(module, 'scale'):
                    module.register_parameter('scale', nn.Parameter(torch.tensor(scale)))
    
    return q_model


def apply_quantization_to_weights(model, weight_bits):
    """
    Apply fake quantization to model weights
    """
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            # Get scale
            if hasattr(model.get_parameter(name), 'scale'):
                scale = model.get_parameter(name).scale
            else:
                # Initialize scale
                scale = param.abs().max() / (2 ** (weight_bits - 1))
            
            # Quantize: round to nearest integer after dividing by scale
            param.data = torch.round(param.data / scale) * scale


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)


def create_checkpoint_dir(checkpoint_dir='./checkpoints'):
    """Create checkpoint directory if it doesn't exist"""
    os.makedirs(checkpoint_dir, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, top1_acc, top5_acc, checkpoint_dir='./checkpoints'):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
    }
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    top1_acc = checkpoint['top1_acc']
    top5_acc = checkpoint['top5_acc']
    
    return model, optimizer, epoch, top1_acc, top5_acc


def count_parameters(model):
    """Count total number of parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(prefer_cuda=True):
    """Get device (CUDA or CPU)"""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def print_model_info(model, model_name='Model'):
    """Print model information"""
    total_params = count_parameters(model)
    print(f"\n{'='*50}")
    print(f"{model_name} Information:")
    print(f"{'='*50}")
    print(f"Total parameters: {total_params:,}")
    print(f"{'='*50}\n")
