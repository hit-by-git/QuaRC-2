# QuaRC Quick Reference Guide

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
python test_implementation.py
```

### 3. Train Model
```bash
python main.py
```

## 📊 Reproduce Paper Results

### Run All Experiments
```bash
python run_experiments.py
```

### Run Specific Configuration
Edit `config.py` and change:
```python
WEIGHT_BITS = 2              # Quantization bitwidth
ACTIVATION_BITS = 32         # Keep full precision for MobileNetV2
CORESET_FRACTION = 0.01      # 1% of dataset
```

Then run:
```bash
python main.py
```

## 📈 Expected Results (Table I from Paper)

### CIFAR-100 + MobileNetV2

| Bits | Data | Top-1 (%) | Top-5 (%) |
|------|------|-----------|-----------|
| 2w32a | 1% | 56.36 ± 0.5 | 85.08 ± 0.3 |
| 2w32a | 5% | 63.30 ± 0.3 | 88.01 ± 0.2 |
| 3w32a | 1% | 68.69 ± 0.4 | 90.89 ± 0.2 |
| 3w32a | 5% | 70.11 ± 0.3 | 91.77 ± 0.2 |
| 4w32a | 1% | 71.25 ± 0.2 | 91.39 ± 0.2 |

## 🔧 Configuration Reference

### Model Settings
```python
MODEL_NAME = "mobilenetv2"      # 'mobilenetv2', 'resnet18'
PRETRAINED = True               # Load ImageNet weights
NUM_CLASSES = 100              # CIFAR-100
```

### Quantization Settings
```python
WEIGHT_BITS = 2                # 2, 3, or 4 bits
ACTIVATION_BITS = 32           # Full precision (keep for MobileNetV2)
SYMMETRIC_QUANTIZATION = True  # Symmetric or asymmetric
```

### Training Parameters
```python
BATCH_SIZE = 256
NUM_EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
OPTIMIZER = "SGD"
MOMENTUM = 0.9
```

### Coreset Selection
```python
CORESET_FRACTION = 0.01        # 1% of training data
SELECTION_INTERVAL = 50        # Reselect every 50 epochs
USE_RES = True                 # Relative Entropy Score
USE_EVS = True                 # Error Vector Score  
USE_DS = True                  # Disagreement Score
```

### Cascaded Layer Correction
```python
USE_CLC = True                 # Enable CLC
CLC_BETA = 1e5                 # Weight for CLC loss
CLC_CORRECT_ALL_LAYERS = True  # Correct all intermediate layers
```

### Knowledge Distillation
```python
USE_KD = True                  # Enable KD
TEACHER_MODEL = "mobilenetv2"
KD_TEMPERATURE = 4.0           # Temperature scaling
```

## 📁 Output Structure

```
QuaRC-2/
├── checkpoints/
│   ├── checkpoint_epoch_10.pt
│   ├── checkpoint_epoch_20.pt
│   └── best_*.pt              # Best model
├── logs/
│   ├── quarc_20240413_153022.log
│   └── experiments_*.log
└── data/
    └── cifar-100/             # CIFAR-100 dataset (auto-downloaded)
```

## 🎯 Key Metrics

### Top-1 Accuracy
Percentage of samples where the predicted class matches the ground truth.

### Top-5 Accuracy  
Percentage of samples where the ground truth is in the top 5 predictions.

### Relative Entropy Score (RES)
Measures how well a sample captures quantization errors:
- Higher RES → More important for training
- Used to select coreset samples

## 🔍 Monitoring Training

Check logs in real-time:
```bash
tail -f logs/quarc_*.log
```

Key metrics logged:
- Training loss
- Top-1 and Top-5 accuracy
- Learning rate schedule
- Coreset selection progress

## 💾 Checkpoints

Checkpoints are saved automatically:
- Every 10 epochs: `checkpoint_epoch_10.pt`
- Best model: `best_*.pt`

Load checkpoint:
```python
from model_utils import load_checkpoint
model, optimizer, epoch, top1, top5 = load_checkpoint(model, optimizer, 'checkpoints/best_*.pt')
```

## ⚙️ GPU Memory Requirements

| Batch Size | GPU Memory | Max Epochs |
|------------|-----------|-----------|
| 256 | ~6 GB | 200 |
| 128 | ~3 GB | 200 |
| 64  | ~2 GB | 200 |
```

If OOM, reduce `BATCH_SIZE` in config.py.

## 🚨 Troubleshooting

### CUDA Out of Memory
```python
BATCH_SIZE = 128  # Reduce from 256
NUM_WORKERS = 2   # Reduce from 4
```

### Slow Data Loading
```python
NUM_WORKERS = 8   # Increase parallel workers
PIN_MEMORY = True # Already set in config
```

### Reproducibility Issues
```python
SEED = 42         # Already set in config
# Also set in environment:
# export CUBLAS_WORKSPACE_CONFIG=:16:8
```

## 📚 Paper Sections & Implementation

| Section | Implementation |
|---------|-----------------|
| III-B (RES) | `coreset_selection.py:calculate_relative_entropy()` |
| III-C (CLC) | `trainer.py:CascadedLayerCorrectionLoss` |
| Algorithm 1 | `main.py:main()` training loop |
| Experiments | `run_experiments.py` |

## 🔗 Related Files

- **Configuration**: `config.py`
- **Data Loading**: `data_loader.py`
- **Models**: `model_utils.py`
- **Quantization**: `quantization.py`
- **Coreset**: `coreset_selection.py`
- **Training**: `trainer.py`
- **Main**: `main.py`
- **Tests**: `test_implementation.py`

## 📞 Debugging Tips

1. **Check device**: Print GPU memory in logs
2. **Verify data**: Inspect batch shapes in data loader
3. **Monitor loss**: Loss should decrease over time
4. **Check scores**: RES values should be positive

## 🎓 Understanding the Algorithm

### Flow
1. **Load Models**: Full-precision (teacher) and quantized (student)
2. **Epoch Loop**:
   - Every 50 epochs: Compute RES and select coreset
   - Train on coreset with KD + CLC losses
   - Evaluate on full test set
3. **Save**: Best model and periodic checkpoints

### Losses
- **KD Loss**: Aligns final outputs between student and teacher
- **CLC Loss**: Aligns intermediate layer outputs
- **Total**: KD + β × CLC (β = 10^5)

## ✅ Validation Checklist

Before running full experiments:
- [ ] Run `test_implementation.py` - all tests pass
- [ ] CIFAR-100 dataset auto-downloads
- [ ] GPU has sufficient memory
- [ ] Output directories created
- [ ] Logs show sensible loss values

## 🎯 Tips for Best Results

1. **Hardware**: Use GPU for ~50x speedup
2. **Data**: First 10 epochs are crucial for initialization
3. **Selection**: RES becomes more accurate after epoch 50
4. **Hyperparameters**: Keep β = 1e5 for CLC
5. **Learning Rate**: Cosine annealing helps convergence

---

**Last Updated**: April 2026
**Version**: 1.0
**Status**: Production Ready ✅
