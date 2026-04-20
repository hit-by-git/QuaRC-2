# QuaRC: Quantization-Aware Training on Edge Devices

Implementation of "Enhancing Quantization-Aware Training on Edge Devices via Relative Entropy Coreset Selection and Cascaded Layer Correction"

## Overview

QuaRC is a framework that enables efficient Quantization-Aware Training (QAT) on edge devices by:

1. **Relative Entropy Score (RES)** - Intelligently selects representative training samples that capture quantization errors
2. **Cascaded Layer Correction (CLC)** - Aligns intermediate layer outputs to reduce quantization error propagation

## Key Features

- ✅ CIFAR-100 dataset support with MobileNetV2
- ✅ LSQ+ (Learned Step size Quantization+) method
- ✅ Knowledge Distillation (KD) for improved accuracy
- ✅ Adaptive coreset selection every R epochs
- ✅ Full reproducibility with configurable hyperparameters
- ✅ Comprehensive logging and checkpointing

## Project Structure

```
QuaRC-2/
├── config.py                 # Configuration and hyperparameters
├── data_loader.py           # CIFAR-100 data loading utilities
├── model_utils.py           # Model loading and utilities
├── quantization.py          # Quantization and fake quantization operations
├── coreset_selection.py     # Relative Entropy Score (RES) implementation
├── trainer.py               # Training loop with KD and CLC losses
├── main.py                  # Main training script
├── run_experiments.py       # Experiment runner for paper reproduction
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

### 1. Clone the Repository

```bash
cd QuaRC-2
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision {torchvision.__version__}')"
```

## Usage

### Quick Start: Run Single Experiment

```bash
python main.py
```

This will train a 2-bit quantized MobileNetV2 on CIFAR-100 with 1% coreset.

### Reproduce Paper Results

Run all experiments from Table I of the paper:

```bash
python run_experiments.py
```

This will sequentially train:
- CIFAR-100 + MobileNetV2 (2w32a) on 1% coreset
- CIFAR-100 + MobileNetV2 (3w32a) on 1% coreset  
- CIFAR-100 + MobileNetV2 (2w32a) on 5% coreset
- CIFAR-100 + MobileNetV2 (3w32a) on 5% coreset

### Custom Configuration

Edit `config.py` to customize:

```python
# Quantization settings
WEIGHT_BITS = 2              # 2, 3, or 4 bits
ACTIVATION_BITS = 32        # Full precision for MobileNetV2
CORESET_FRACTION = 0.01     # 1% of dataset

# Training parameters
BATCH_SIZE = 256
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4

# CLC settings
CLC_BETA = 1e5              # Weight for CLC loss
USE_CLC = True
USE_KD = True

# Coreset selection
SELECTION_INTERVAL = 50     # Reselect every 50 epochs
USE_RES = True              # Relative Entropy Score
USE_EVS = True              # Error Vector Score
USE_DS = True               # Disagreement Score
```

Then run:

```bash
python main.py
```

## Expected Results

### CIFAR-100 + MobileNetV2 (Table I from paper)

| Bit Width | Coreset | Top-1 (%) | Top-5 (%) |
|-----------|---------|-----------|-----------|
| 2w32a | 1% | 56.36 | 85.08 |
| 2w32a | 5% | 63.30 | 88.01 |
| 3w32a | 1% | 68.69 | 90.89 |
| 3w32a | 5% | 70.11 | 91.77 |
| 4w32a | 1% | 71.25 | 91.39 |

## Algorithm Details

### 1. Relative Entropy Score (RES)

Measures how well a sample captures quantization errors:

$$d_{RES} = \sum_m p_Q(w_q, x_q) \log\left(\frac{p_Q(w_q, x_q)}{p_F(w_r, x_r)}\right)$$

Where:
- $p_Q$: Output distribution from quantized model
- $p_F$: Output distribution from full-precision model
- Higher RES → Sample is more informative for reducing quantization errors

### 2. Coreset Selection Metric

Combines three metrics with cosine annealing weight:

$$d_S(t) = \alpha(t) \cdot d_{EVS}(t) + (1-\alpha(t)) \cdot d_{DS}(t) + d_{RES}(t)$$

Where:
- $\alpha(t) = \cos\left(\frac{t}{2T}\pi\right)$: Cosine annealing weight
- $d_{EVS}$: Error Vector Score (distance from ground truth)
- $d_{DS}$: Disagreement Score (distance between outputs)
- $d_{RES}$: Relative Entropy Score

### 3. Cascaded Layer Correction (CLC)

Aligns intermediate layer outputs to reduce error propagation:

$$L_{CLC} = \sum_c p_Q^c(w_q, x_q) \log\left(\frac{p_Q^c(w_q, x_q)}{p_F^c(w_r, x_r)}\right)$$

### 4. Total Training Loss

$$L_{TOTAL} = L_{KD} + \beta \cdot L_{CLC}$$

Where:
- $L_{KD}$: Knowledge Distillation loss
- $\beta = 10^5$: Weight for CLC loss

## Training Details

**Data Augmentation:**
- RandomCrop(32, padding=4)
- RandomHorizontalFlip
- Normalization: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]

**Optimizer:**
- SGD with momentum=0.9
- Learning rate: 0.01
- Weight decay: 5e-4
- Cosine annealing schedule over 200 epochs

**Coreset Selection:**
- Performed every R=50 epochs
- Top S% samples selected (S=1, 5, 10, etc.)

## Output Files

- **Logs:** `./logs/quarc_*.log` - Training logs and metrics
- **Checkpoints:** `./checkpoints/checkpoint_epoch_*.pt` - Model checkpoints
- **Best Model:** `./checkpoints/best_*.pt` - Best performing model

## Troubleshooting

### Out of Memory

If you get CUDA out of memory errors:
```python
# In config.py
BATCH_SIZE = 128  # Reduce batch size
```

### Slow Training

Use multiple workers for data loading:
```python
NUM_WORKERS = 8  # Increase number of workers
```

### Reproducibility

Ensure results are reproducible:
```python
SEED = 42  # Set in config.py
```

## Performance Tips

1. **GPU Acceleration:** Use CUDA for ~50x speedup
2. **Multiple GPUs:** Modify main.py to use DataParallel
3. **Mixed Precision:** Can be added to reduce memory usage
4. **Gradient Accumulation:** For larger effective batch sizes

## References

**Original Paper:**
```
Enhancing Quantization-Aware Training on Edge Devices via Relative Entropy 
Coreset Selection and Cascaded Layer Correction

Yujia Tong, Jingling Yuan, Chuang Hu
IEEE Transactions on Mobile Computing, 2026
```

**Related Work:**
- LSQ+: Improving low-bit quantization through learnable offsets and better initialization
- Knowledge Distillation for Quantization
- Coreset Selection for Efficient Learning

## Citation

If you use this code in your research, please cite:

```bibtex
@article{tong2026quarc,
  title={Enhancing Quantization-Aware Training on Edge Devices via Relative Entropy 
         Coreset Selection and Cascaded Layer Correction},
  author={Tong, Yujia and Yuan, Jingling and Hu, Chuang},
  journal={IEEE Transactions on Mobile Computing},
  year={2026}
}
```

## License

This implementation is provided for research and educational purposes.

## Contact & Support

For issues or questions, please refer to the paper or contact the authors.

---

**Last Updated:** April 2026
**Status:** Implementation Complete ✅
