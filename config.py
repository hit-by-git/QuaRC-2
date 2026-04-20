"""
Configuration for QuaRC - Quantization-Aware Training with Relative Entropy Coreset Selection
and Cascaded Layer Correction
Based on the paper: "Enhancing Quantization-Aware Training on Edge Devices via Relative Entropy
Coreset Selection and Cascaded Layer Correction"
"""

# Dataset configuration
DATASET = "cifar100"
DATASET_PATH = "./data"
NUM_CLASSES = 100

# Model configuration
MODEL_NAME = "mobilenetv2"
PRETRAINED = True

# Quantization configuration
QUANTIZATION_METHOD = "LSQ+"  # Learned Step size Quantization+
WEIGHT_BITS = 2  # or 3, 4
ACTIVATION_BITS = 32  # Full precision for MobileNetV2
SYMMETRIC_QUANTIZATION = True

# Training configuration
BATCH_SIZE = 256
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
OPTIMIZER = "SGD"
MOMENTUM = 0.9

# Coreset selection configuration
CORESET_FRACTION = 0.01  # 1% of dataset
SELECTION_INTERVAL = 50  # Perform selection every R=50 epochs
USE_RELATIVE_ENTROPY = True

# Cascaded Layer Correction configuration
USE_CLC = True
CLC_BETA = 1e5  # Weight for CLC loss
CLC_CORRECT_ALL_LAYERS = True  # Correct all intermediate layers

# Knowledge Distillation configuration
USE_KD = True
TEACHER_MODEL = "mobilenetv2"
TEACHER_PRETRAINED = True
KD_TEMPERATURE = 4.0

# Metrics for coreset selection
USE_EVS = True  # Error Vector Score
USE_DS = True   # Disagreement Score
USE_RES = True  # Relative Entropy Score
USE_COSINE_ANNEALING = True  # Cosine annealing weight for metric combination

# Hardware configuration
DEVICE = "cuda"
NUM_WORKERS = 4
PIN_MEMORY = True

# Checkpoint and logging
CHECKPOINT_DIR = "./checkpoints"
LOG_DIR = "./logs"
SAVE_FREQUENCY = 10  # Save checkpoint every 10 epochs
LOG_FREQUENCY = 50   # Log metrics every 50 batches

# Random seed for reproducibility
SEED = 42
