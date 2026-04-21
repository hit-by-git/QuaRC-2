"""
Main training script for QuaRC
Quantization-Aware Training with Relative Entropy Coreset Selection and Cascaded Layer Correction

# QuaRC pipeline
# 1. LSQ+ quantization setup
# 2. RES / EVS / DS coreset scoring
# 3. Supervised + KD + CLC training loss
# 4. Evaluation and checkpointing
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


@torch.no_grad()
def evaluate_classifier(model, data_loader, device):
    """Evaluate Top-1/Top-5 for a standard classifier."""
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    for inputs, targets in tqdm(data_loader, desc="Teacher eval", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _, pred = outputs.max(1)
        top1_correct += pred.eq(targets).sum().item()

        _, top5_pred = outputs.topk(5, 1, True, True)
        top5_pred = top5_pred.t()
        correct = top5_pred.eq(targets.view(1, -1).expand_as(top5_pred))
        top5_correct += correct[:5].reshape(-1).float().sum(0, keepdim=True).item()

        total += targets.size(0)

    top1_acc = 100.0 * top1_correct / total
    top5_acc = 100.0 * top5_correct / total
    return top1_acc, top5_acc


def maybe_prepare_teacher(logger, train_loader, test_loader, device):
    """Load a CIFAR-100 teacher checkpoint or train it if missing."""
    teacher_model = get_model(
        TEACHER_MODEL,
        num_classes=NUM_CLASSES,
        pretrained=TEACHER_PRETRAINED,
        device=device,
    )

    teacher_ckpt = TEACHER_CHECKPOINT_PATH
    teacher_dir = os.path.dirname(teacher_ckpt)
    if teacher_dir:
        os.makedirs(teacher_dir, exist_ok=True)

    should_train_teacher = False

    if os.path.exists(teacher_ckpt):
        state = torch.load(teacher_ckpt, map_location=device)
        teacher_model.load_state_dict(state["model_state_dict"])
        logger.info(f"Loaded CIFAR-100 teacher checkpoint from {teacher_ckpt}")
        top1_acc, top5_acc = evaluate_classifier(teacher_model, test_loader, device)
        logger.info(f"Loaded teacher quality - Top-1: {top1_acc:.2f}% Top-5: {top5_acc:.2f}%")
        if top1_acc < MIN_TEACHER_TOP1:
            logger.warning(
                f"Teacher Top-1 ({top1_acc:.2f}%) is below MIN_TEACHER_TOP1 ({MIN_TEACHER_TOP1:.2f}%). "
                "Retraining teacher."
            )
            should_train_teacher = True
    elif PRETRAIN_TEACHER_IF_MISSING:
        should_train_teacher = True

    if should_train_teacher and PRETRAIN_TEACHER_IF_MISSING:
        logger.info("Teacher checkpoint not found. Pretraining teacher on CIFAR-100...")
        optimizer = optim.SGD(
            teacher_model.parameters(),
            lr=TEACHER_LEARNING_RATE,
            momentum=MOMENTUM,
            weight_decay=TEACHER_WEIGHT_DECAY,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TEACHER_EPOCHS)
        criterion = nn.CrossEntropyLoss()

        best_top1 = 0.0
        best_top5 = 0.0

        for epoch in range(TEACHER_EPOCHS):
            teacher_model.train()
            running_loss = 0.0
            total_samples = 0

            pbar = tqdm(train_loader, desc=f"Teacher epoch {epoch+1}/{TEACHER_EPOCHS}")
            for inputs, targets in pbar:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = teacher_model(inputs)
                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
                pbar.set_postfix({"loss": f"{running_loss / max(total_samples, 1):.4f}"})

            scheduler.step()
            top1_acc, top5_acc = evaluate_classifier(teacher_model, test_loader, device)
            logger.info(
                f"Teacher epoch {epoch+1}/{TEACHER_EPOCHS} - "
                f"Top-1: {top1_acc:.2f}% Top-5: {top5_acc:.2f}%"
            )

            if top1_acc > best_top1:
                best_top1 = top1_acc
                best_top5 = top5_acc
                torch.save(
                    {
                        "model_state_dict": teacher_model.state_dict(),
                        "top1_acc": best_top1,
                        "top5_acc": best_top5,
                    },
                    teacher_ckpt,
                )

        logger.info(
            f"Saved teacher checkpoint to {teacher_ckpt} "
            f"(Top-1: {best_top1:.2f}% Top-5: {best_top5:.2f}%)"
        )

        best_state = torch.load(teacher_ckpt, map_location=device)
        teacher_model.load_state_dict(best_state["model_state_dict"])
    elif not os.path.exists(teacher_ckpt):
        logger.warning(
            "Teacher checkpoint missing and PRETRAIN_TEACHER_IF_MISSING=False. "
            "KD/CLC quality may be weak."
        )

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    return teacher_model


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
    
    # QuaRC Step 1 and Step 2: load CIFAR-100 and prepare coreset scoring data
    logger.info("Loading CIFAR-100 dataset...")
    full_dataset = get_full_dataset(split='train')
    test_train_loader, test_loader, _, test_dataset = get_cifar100_loaders(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    logger.info(f"Full dataset size: {len(full_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    # QuaRC Step 0: build a CIFAR-100 teacher and quantized student
    logger.info("\nPreparing teacher model...")
    fp_model = maybe_prepare_teacher(logger, test_train_loader, test_loader, device)
    q_model = quantize_model(fp_model, WEIGHT_BITS, ACTIVATION_BITS)

    print_model_info(fp_model, "Full-Precision Teacher")
    print_model_info(q_model, "Quantized Student")

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
    
    # Initialize trainer
    logger.info("\nInitializing trainer...")
    trainer = QuantizationAwareTrainer(
        fp_model=fp_model,
        q_model=q_model,
        optimizer=optimizer,
        device=device,
        use_kd=USE_KD,
        use_clc=USE_CLC,
        clc_beta=CLC_BETA,
        kd_temperature=KD_TEMPERATURE,
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
        
        # QuaRC Step 2: adaptive coreset selection from Equation 6
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
        
        # QuaRC Step 3: optimize CE + KD + CLC loss
        avg_loss = trainer.train_epoch(train_loader, epoch, log_frequency=LOG_FREQUENCY)
        logger.info(f"Average Loss: {avg_loss:.4f}")
        
        # QuaRC Step 4: evaluate Top-1 / Top-5 metrics
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
