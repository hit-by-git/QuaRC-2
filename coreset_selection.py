"""
Coreset Selection with Relative Entropy Score (RES)
Based on Algorithm 1 from the QuaRC paper

# QuaRC Step 2: Sample scoring
# - Relative Entropy Score (RES)
# - Error Vector Score (EVS)
# - Disagreement Score (DS)
# - Equation 6: final coreset score
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import math


def calculate_relative_entropy(p_q, p_f, reduction='mean'):
    """
    Calculate Relative Entropy Score (RES) between quantized and full-precision outputs
    
    RES = sum_m p_Q(w_q, x_q) * log(p_Q(w_q, x_q) / p_F(w_r, x_r))
    
    Args:
        p_q: Output logits from quantized model (batch_size, num_classes)
        p_f: Output logits from full-precision model (batch_size, num_classes)
        reduction: 'mean' or 'sum'
    
    Returns:
        Relative entropy for each sample in batch
    """
    # QuaRC formula (RES):
    # d_RES = sum_m p_Q(w_q, x_q) * log(p_Q(w_q, x_q) / p_F(w_r, x_r))
    # Convert logits to probabilities
    p_q_softmax = F.softmax(p_q, dim=1)
    p_f_softmax = F.softmax(p_f, dim=1)
    
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-10
    p_q_safe = torch.clamp(p_q_softmax, min=epsilon)
    p_f_safe = torch.clamp(p_f_softmax, min=epsilon)
    
    # Calculate relative entropy: sum_m p_Q * log(p_Q / p_F)
    # Shape: (batch_size, num_classes)
    ratio = p_q_safe / p_f_safe
    entropy_per_class = p_q_safe * torch.log(ratio)
    
    # Sum over classes to get per-sample entropy
    # Shape: (batch_size,)
    d_res = entropy_per_class.sum(dim=1)
    
    if reduction == 'mean':
        return d_res.mean()
    elif reduction == 'none':
        return d_res
    else:
        return d_res.sum()


def calculate_error_vector_score(p_q, targets, reduction='mean'):
    """
    Calculate Error Vector Score (EVS) - L2 distance from ground truth
    
    d_EVS = ||p_Q(w_q, x) - y||_2
    
    Args:
        p_q: Output logits from quantized model
        targets: Ground truth labels (one-hot or class indices)
        reduction: 'mean' or 'none'
    
    Returns:
        Error vector score for each sample
    """
    # Convert targets to one-hot if needed
    if targets.dim() == 1:
        num_classes = p_q.size(1)
        targets_onehot = torch.zeros_like(p_q)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
    else:
        targets_onehot = targets
    
    # L2 distance
    # Shape: (batch_size,)
    d_evs = torch.norm(p_q - targets_onehot, dim=1, p=2)
    
    if reduction == 'mean':
        return d_evs.mean()
    elif reduction == 'none':
        return d_evs
    else:
        return d_evs.sum()


def calculate_disagreement_score(p_q, p_f, reduction='mean'):
    """
    Calculate Disagreement Score (DS) - L2 distance between outputs
    
    d_DS = ||p_Q(w_q, x) - p_T(w_r, x)||_2
    
    Args:
        p_q: Output logits from quantized model
        p_f: Output logits from full-precision model
        reduction: 'mean' or 'none'
    
    Returns:
        Disagreement score for each sample
    """
    # L2 distance between outputs
    # Shape: (batch_size,)
    d_ds = torch.norm(p_q - p_f, dim=1, p=2)
    
    if reduction == 'mean':
        return d_ds.mean()
    elif reduction == 'none':
        return d_ds
    else:
        return d_ds.sum()


def calculate_cosine_annealing_weight(epoch, total_epochs):
    """
    Cosine annealing schedule for weight coefficient
    
    alpha(t) = cos(t / (2*T) * pi)
    
    Args:
        epoch: Current epoch
        total_epochs: Total number of epochs
    
    Returns:
        Weight coefficient in [0, 1]
    """
    alpha = math.cos((epoch / (2 * total_epochs)) * math.pi)
    return alpha


def compute_coreset_scores(fp_model, q_model, train_loader, epoch, total_epochs, 
                          device, use_evs=True, use_ds=True, use_res=True,
                          use_cosine_annealing=True):
    """
    Compute coreset selection scores for all samples
    
    Final metric: d_S(t) = alpha(t)*d_EVS(t) + (1-alpha(t))*d_DS(t) + d_RES(t)

    # Equation 6 from the QuaRC paper:
    # d_S(t) = alpha(t) * d_EVS(t) + (1 - alpha(t)) * d_DS(t) + d_RES(t)
    # alpha(t) = cos(t / (2T) * pi)
    
    Args:
        fp_model: Full-precision model
        q_model: Quantized model
        train_loader: DataLoader for training data
        epoch: Current epoch
        total_epochs: Total number of epochs
        device: Device to run computation on
        use_evs: Whether to use error vector score
        use_ds: Whether to use disagreement score
        use_res: Whether to use relative entropy score
        use_cosine_annealing: Whether to use cosine annealing for weight
    
    Returns:
        scores: Array of scores for each sample in dataset
        all_indices: Indices corresponding to scores
    """
    fp_model.eval()
    q_model.eval()
    
    all_scores_evs = []
    all_scores_ds = []
    all_scores_res = []
    all_indices = []
    
    if use_cosine_annealing:
        alpha = calculate_cosine_annealing_weight(epoch, total_epochs)
    else:
        alpha = 0.5  # Default equal weighting
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc="Computing scores")):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get outputs from both models
            p_f = fp_model(inputs)
            p_q = q_model(inputs)
            
            # Calculate individual scores
            if use_evs:
                d_evs = calculate_error_vector_score(p_q, targets, reduction='none')
                all_scores_evs.append(d_evs.cpu().numpy())
            
            if use_ds:
                d_ds = calculate_disagreement_score(p_q, p_f, reduction='none')
                all_scores_ds.append(d_ds.cpu().numpy())
            
            if use_res:
                d_res = calculate_relative_entropy(p_q, p_f, reduction='none')
                all_scores_res.append(d_res.cpu().numpy())
            
            # Track indices
            batch_indices = np.arange(batch_idx * train_loader.batch_size,
                                     min((batch_idx + 1) * train_loader.batch_size,
                                         len(train_loader.dataset)))
            all_indices.append(batch_indices)
    
    # Concatenate all scores
    all_indices = np.concatenate(all_indices)
    
    if use_evs:
        all_scores_evs = np.concatenate(all_scores_evs)
    else:
        all_scores_evs = np.zeros(len(all_indices))
    
    if use_ds:
        all_scores_ds = np.concatenate(all_scores_ds)
    else:
        all_scores_ds = np.zeros(len(all_indices))
    
    if use_res:
        all_scores_res = np.concatenate(all_scores_res)
    else:
        all_scores_res = np.zeros(len(all_indices))
    
    # Combine scores: d_S(t) = alpha(t)*d_EVS(t) + (1-alpha(t))*d_DS(t) + d_RES(t)
    combined_scores = alpha * all_scores_evs + (1 - alpha) * all_scores_ds + all_scores_res
    
    return combined_scores, all_indices


def select_coreset(scores, indices, coreset_fraction):
    """
    Select top S% samples based on scores
    
    Args:
        scores: Score for each sample
        indices: Original indices of samples
        coreset_fraction: Fraction of samples to select (e.g., 0.1 for 10%)
    
    Returns:
        coreset_indices: Indices of selected samples in original dataset
    """
    # Sort by score in descending order
    sorted_idx = np.argsort(-scores)  # Negative for descending
    
    # Select top S% samples
    n_select = max(1, int(len(scores) * coreset_fraction))
    selected_sorted_idx = sorted_idx[:n_select]
    
    # Get original indices
    coreset_indices = indices[selected_sorted_idx]
    
    return coreset_indices.astype(int)


def adaptive_coreset_selection(fp_model, q_model, full_dataset, epoch, total_epochs,
                              device, coreset_fraction=0.1, batch_size=256,
                              num_workers=4, use_evs=True, use_ds=True, 
                              use_res=True, use_cosine_annealing=True):
    """
    Perform adaptive coreset selection every R epochs
    
    Args:
        fp_model: Full-precision model
        q_model: Quantized model
        full_dataset: Full training dataset
        epoch: Current epoch
        total_epochs: Total number of epochs
        device: Device for computation
        coreset_fraction: Fraction of data to select
        batch_size: Batch size for computation
        num_workers: Number of workers
        use_evs: Use EVS metric
        use_ds: Use DS metric
        use_res: Use RES metric
        use_cosine_annealing: Use cosine annealing
    
    Returns:
        coreset_indices: Indices for selected coreset
        coreset_dataset: Subset dataset for training
    """
    # Create loader for full dataset
    full_loader = DataLoader(
        full_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Compute scores for all samples
    scores, indices = compute_coreset_scores(
        fp_model, q_model, full_loader, epoch, total_epochs, device,
        use_evs=use_evs, use_ds=use_ds, use_res=use_res,
        use_cosine_annealing=use_cosine_annealing
    )
    
    # Select coreset
    coreset_indices = select_coreset(scores, indices, coreset_fraction)
    
    # Create subset dataset
    from torch.utils.data import Subset
    coreset_dataset = Subset(full_dataset, coreset_indices)
    
    return coreset_indices, coreset_dataset
