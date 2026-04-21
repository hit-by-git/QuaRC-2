"""
Training utilities with Knowledge Distillation (KD) and Cascaded Layer Correction (CLC)

# QuaRC Step 3: Training objective
# - Supervised cross-entropy loss on CIFAR-100 labels
# - Knowledge Distillation loss
# - Cascaded Layer Correction loss
# - Final objective: L_TOTAL = L_CE + L_KD + beta * L_CLC
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss
    
    L_KD = -sum_m p_T(w_r, x_i) * log(p_Q(w_q, x_i))
    """
    
    def __init__(self, temperature=4.0):
        super().__init__()
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, student_logits, teacher_logits):
        """
        Args:
            student_logits: Output from quantized model
            teacher_logits: Output from full-precision model
        
        Returns:
            KD loss
        """
        # Apply temperature scaling
        student_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # KL divergence
        loss = self.kl_div(student_probs, teacher_probs)
        
        # Scale by temperature squared (as in standard KD formulation)
        loss = loss * (self.temperature ** 2)
        
        return loss


class CascadedLayerCorrectionLoss(nn.Module):
    """
    Cascaded Layer Correction (CLC) Loss
    
    L_CLC = sum_c p_Q^c(w_q, x_q) * log(p_Q^c(w_q, x_q) / p_F^c(w_r, x_r))
    
    where c represents intermediate layers to be optimized
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, student_intermediate, teacher_intermediate):
        """
        Args:
            student_intermediate: Output from intermediate layer of quantized model
            teacher_intermediate: Output from intermediate layer of full-precision model
        
        Returns:
            CLC loss (KL divergence between intermediate outputs)
        """
        # QuaRC formula: L_CLC = sum_c p_Q^c(w_q, x_q) * log(p_Q^c(w_q, x_q) / p_F^c(w_r, x_r))
        # Reshape if necessary (flatten for FC layers, keep spatial for conv)
        if student_intermediate.dim() > 2:
            # For conv layers, use spatial average
            student_flat = F.adaptive_avg_pool2d(student_intermediate, 1).view(student_intermediate.size(0), -1)
            teacher_flat = F.adaptive_avg_pool2d(teacher_intermediate, 1).view(teacher_intermediate.size(0), -1)
        else:
            student_flat = student_intermediate
            teacher_flat = teacher_intermediate
        
        # Convert to probabilities
        epsilon = 1e-10
        student_probs = torch.clamp(F.softmax(student_flat, dim=1), min=epsilon)
        teacher_probs = torch.clamp(F.softmax(teacher_flat, dim=1), min=epsilon)
        
        # KL divergence
        kl_loss = F.kl_div(
            torch.log(student_probs),
            teacher_probs,
            reduction='batchmean'
        )
        
        return kl_loss


def extract_intermediate_features(model, inputs, layer_name=None):
    """
    Extract intermediate layer features from model
    
    Args:
        model: PyTorch model
        inputs: Input tensor
        layer_name: Name of layer to extract (optional)
    
    Returns:
        Dictionary of intermediate features
    """
    features = {}
    handles = []
    
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hooks for specific layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            handle = module.register_forward_hook(get_activation(name))
            handles.append(handle)
    
    # Forward pass
    with torch.no_grad():
        _ = model(inputs)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return features


class QuantizationAwareTrainer:
    """
    Trainer for Quantization-Aware Training with KD and CLC
    """
    
    def __init__(self, fp_model, q_model, optimizer, device, 
                 use_kd=True, use_clc=True, clc_beta=1e5,
                 kd_temperature=4.0):
        """
        Args:
            fp_model: Full-precision teacher model
            q_model: Quantized student model
            optimizer: Optimizer for quantized model
            device: Device for computation
            use_kd: Whether to use knowledge distillation
            use_clc: Whether to use cascaded layer correction
            clc_beta: Weight for CLC loss
            kd_temperature: Temperature for KD
        """
        self.fp_model = fp_model
        self.q_model = q_model
        self.optimizer = optimizer
        self.device = device
        self.use_kd = use_kd
        self.use_clc = use_clc
        self.clc_beta = clc_beta
        
        # Loss functions
        self.kd_loss = KnowledgeDistillationLoss(temperature=kd_temperature)
        self.clc_loss = CascadedLayerCorrectionLoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, epoch, log_frequency=50):
        """
        Train for one epoch
        
        Args:
            train_loader: DataLoader for training data
            epoch: Current epoch number
            log_frequency: How often to print metrics
        
        Returns:
            Average loss for the epoch
        """
        self.q_model.train()
        self.fp_model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass for the student; gradients must flow here
            q_logits = self.q_model(inputs)
            
            with torch.no_grad():
                fp_logits = self.fp_model(inputs)
            
            # QuaRC Step 3: Final objective
            # L_TOTAL = L_CE + L_KD + beta * L_CLC
            loss_ce = self.ce_loss(q_logits, targets)
            loss = loss_ce
            loss_kd = torch.tensor(0.0, device=self.device)
            loss_clc = torch.tensor(0.0, device=self.device)
            
            # Knowledge Distillation Loss
            if self.use_kd:
                loss_kd = self.kd_loss(q_logits, fp_logits)
                loss += loss_kd
            
            # Cascaded Layer Correction Loss
            if self.use_clc:
                loss_clc = self._compute_clc_loss(inputs)
                loss += self.clc_beta * loss_clc
            
            # Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Log
            if (batch_idx + 1) % log_frequency == 0:
                avg_loss = total_loss / total_samples
                pbar.set_postfix({
                    'loss': f'{avg_loss:.4f}',
                    'ce': f'{loss_ce.item():.3f}',
                    'kd': f'{loss_kd.item():.3f}',
                    'clc': f'{loss_clc.item():.3f}'
                })
        
        avg_loss = total_loss / total_samples
        return avg_loss
    
    def _compute_clc_loss(self, inputs):
        """
        Compute Cascaded Layer Correction loss
        Aligns intermediate layer outputs
        """
        loss_clc = 0.0
        n_layers = 0
        
        # Get intermediate features from both models
        q_model_hooks = {}
        fp_model_hooks = {}
        
        handles_q = []
        handles_fp = []
        
        def get_activation(target_dict, name):
            def hook(model, input, output):
                target_dict[name] = output
            return hook
        
        # Register hooks for intermediate layers
        for name, module in self.q_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name.count('.') >= 2:
                handle = module.register_forward_hook(get_activation(q_model_hooks, name))
                handles_q.append(handle)
        
        for name, module in self.fp_model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and name.count('.') >= 2:
                handle = module.register_forward_hook(get_activation(fp_model_hooks, name))
                handles_fp.append(handle)
        
        # Forward pass: student must keep gradients; teacher stays frozen
        _ = self.q_model(inputs)
        with torch.no_grad():
            _ = self.fp_model(inputs)
        
        # Compute CLC loss for aligned layers
        for name in q_model_hooks:
            if name in fp_model_hooks:
                q_feat = q_model_hooks[name]
                fp_feat = fp_model_hooks[name]
                
                loss_clc += self.clc_loss(q_feat, fp_feat)
                n_layers += 1
        
        # Remove hooks
        for handle in handles_q + handles_fp:
            handle.remove()
        
        if n_layers > 0:
            loss_clc = loss_clc / n_layers
        
        return loss_clc
    
    @torch.no_grad()
    def evaluate(self, val_loader):
        """
        Evaluate model on validation set
        
        Args:
            val_loader: DataLoader for validation data
        
        Returns:
            Tuple of (top1_accuracy, top5_accuracy)
        """
        self.q_model.eval()
        
        top1_correct = 0
        top5_correct = 0
        total = 0
        
        for inputs, targets in tqdm(val_loader, desc="Evaluating"):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.q_model(inputs)
            
            # Top-1 accuracy
            _, pred = outputs.max(1)
            top1_correct += pred.eq(targets).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_pred = top5_pred.t()
            correct = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))
            top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()
            
            total += targets.size(0)
        
        top1_acc = 100.0 * top1_correct / total
        top5_acc = 100.0 * top5_correct / total
        
        return top1_acc, top5_acc
