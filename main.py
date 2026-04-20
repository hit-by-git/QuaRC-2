"""
Main training script for QuaRC
Quantization-Aware Training with Relative Entropy Coreset Selection and Cascaded Layer Correction
"""
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import logging
from datetime import datetime

# Import modules
from config import *
from data_loader import get_cifar100_loaders, get_full_dataset
from model_utils import (
    get_model, quantize_model, set_seed, create_checkpoint_dir,
    save_checkpoint, get_device, print_model_info
)
from trainer import QuantizationAwareTrainer
from coreset_selection import adaptive_coreset_selection


# Setup logging
def setup_logging(log_dir='./logs'):
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'quarc_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def main():
    """Main training function"""
    
    # Setup
    logger = setup_logging(LOG_DIR)
    set_seed(SEED)
    device = get_device()
    create_checkpoint_dir(CHECKPOINT_DIR)
    
    logger.info("="*60)
    logger.info("QuaRC: Quantization-Aware Training with Coreset Selection")
    logger.info("="*60)
    
    # Log configuration
    logger.info(f"Device: {device}")
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Quantization: {WEIGHT_BITS}-bit weights, {ACTIVATION_BITS}-bit activations")
    logger.info(f"Coreset Fraction: {CORESET_FRACTION*100:.1f}%")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"CLC Beta: {CLC_BETA}")
    
    # Load models
    logger.info("\nLoading models...")
    fp_model = get_model(MODEL_NAME, num_classes=NUM_CLASSES, 
                         pretrained=PRETRAINED, device=device)
    
    q_model = quantize_model(fp_model, WEIGHT_BITS, ACTIVATION_BITS)
    
    print_model_info(fp_model, "Full-Precision Model")
    print_model_info(q_model, "Quantized Model")
    
    # Setup optimizer
    logger.info("Setting up optimizer...")
    optimizer = optim.SGD(
        q_model.parameters(),
        lr=LEARNING_RATE,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )
    
    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    # Load data
    logger.info("Loading CIFAR-100 dataset...")
    full_dataset = get_full_dataset(split='train')
    test_train_loader, test_loader, _, test_dataset = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = QuantizationAwareTrainer(
        fp_model=fp_model,
        q_model=q_model,
        optimizer=optimizer,
        device=device,
        use_kd=USE_KD,
        use_clc=USE_CLC,
        clc_beta=CLC_BETA
    )
    
    # Training loop
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60)
    
    best_top1_acc = 0.0
    best_top5_acc = 0.0
    best_epoch = 0
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Perform coreset selection every R epochs
        if (epoch % SELECTION_INTERVAL == 0) and epoch > 0:
            logger.info("Performing coreset selection...")
            
            try:
                coreset_indices, coreset_dataset = adaptive_coreset_selection(
                    fp_model=fp_model,
                    q_model=q_model,
                    full_dataset=full_dataset,
                    epoch=epoch,
                    total_epochs=NUM_EPOCHS,
                    device=device,
                    coreset_fraction=CORESET_FRACTION,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    use_evs=USE_EVS,
                    use_ds=USE_DS,
                    use_res=USE_RES,
                    use_cosine_annealing=USE_COSINE_ANNEALING
                )
                
                logger.info(f"Coreset size: {len(coreset_indices)} samples ({len(coreset_indices)/len(full_dataset)*100:.1f}%)")
                
                # Create new train loader with coreset
                train_loader, _, _, _ = get_cifar100_loaders(
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    coreset_indices=coreset_indices
                )
            except Exception as e:
                logger.warning(f"Coreset selection failed: {e}. Using full dataset.")
                train_loader = test_train_loader
        else:
            # Use full or sampled dataset for initial epochs
            if epoch == 0:
                # For first epoch, use full dataset to initialize
                train_loader = test_train_loader
            else:
                train_loader = test_train_loader
        
        # Train epoch
        avg_loss = trainer.train_epoch(train_loader, epoch, log_frequency=LOG_FREQUENCY)
        logger.info(f"Average Loss: {avg_loss:.4f}")
        
        # Evaluate
        logger.info("Evaluating on test set...")
        top1_acc, top5_acc = trainer.evaluate(test_loader)
        
        logger.info(f"Top-1 Accuracy: {top1_acc:.2f}%")
        logger.info(f"Top-5 Accuracy: {top5_acc:.2f}%")
        
        # Save checkpoint
        if (epoch + 1) % SAVE_FREQUENCY == 0:
            save_checkpoint(q_model, optimizer, epoch + 1, top1_acc, top5_acc, CHECKPOINT_DIR)
        
        # Track best accuracy
        if top1_acc > best_top1_acc:
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc
            best_epoch = epoch + 1
            # Save best model
            save_checkpoint(q_model, optimizer, f"best_{epoch + 1}", top1_acc, top5_acc, CHECKPOINT_DIR)
        
        # Step scheduler
        scheduler.step()
    
    # Final summary
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)
    logger.info(f"Best Top-1 Accuracy: {best_top1_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"Best Top-5 Accuracy: {best_top5_acc:.2f}%")
    logger.info("="*60)
    
    return best_top1_acc, best_top5_acc


if __name__ == '__main__':
    main()
