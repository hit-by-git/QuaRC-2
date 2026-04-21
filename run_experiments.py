"""
Experiment runner for specific configurations
Reproduces the exact results from the QuaRC paper
"""
import os
import sys
import torch
from datetime import datetime
import logging

# Setup config for each experiment
class ExperimentConfig:
    """Configuration for specific experiments from the paper"""
    
    # CIFAR-100 + MobileNetV2 experiments (Table I)
    CIFAR100_MOBILENETV2_2BIT_1PERCENT = {
        'name': 'CIFAR100_MobileNetV2_2w32a_1%',
        'dataset': 'cifar100',
        'model': 'mobilenetv2',
        'weight_bits': 2,
        'activation_bits': 32,
        'coreset_fraction': 0.01,
        'batch_size': 256,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'selection_interval': 50,
        'clc_beta': 1e5,
        'expected_top1': 56.36,
        'expected_top5': 85.08,
    }
    
    CIFAR100_MOBILENETV2_3BIT_1PERCENT = {
        'name': 'CIFAR100_MobileNetV2_3w32a_1%',
        'dataset': 'cifar100',
        'model': 'mobilenetv2',
        'weight_bits': 3,
        'activation_bits': 32,
        'coreset_fraction': 0.01,
        'batch_size': 256,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'selection_interval': 50,
        'clc_beta': 1e5,
        'expected_top1': 68.69,
        'expected_top5': 90.89,
    }
    
    CIFAR100_MOBILENETV2_2BIT_5PERCENT = {
        'name': 'CIFAR100_MobileNetV2_2w32a_5%',
        'dataset': 'cifar100',
        'model': 'mobilenetv2',
        'weight_bits': 2,
        'activation_bits': 32,
        'coreset_fraction': 0.05,
        'batch_size': 256,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'selection_interval': 50,
        'clc_beta': 1e5,
        'expected_top1': 63.30,
        'expected_top5': 88.01,
    }
    
    CIFAR100_MOBILENETV2_3BIT_5PERCENT = {
        'name': 'CIFAR100_MobileNetV2_3w32a_5%',
        'dataset': 'cifar100',
        'model': 'mobilenetv2',
        'weight_bits': 3,
        'activation_bits': 32,
        'coreset_fraction': 0.05,
        'batch_size': 256,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'selection_interval': 50,
        'clc_beta': 1e5,
        'expected_top1': 70.11,
        'expected_top5': 91.77,
    }

    CIFAR100_MOBILENETV2_4BIT_1PERCENT = {
        'name': 'CIFAR100_MobileNetV2_4w32a_1%',
        'dataset': 'cifar100',
        'model': 'mobilenetv2',
        'weight_bits': 4,
        'activation_bits': 32,
        'coreset_fraction': 0.01,
        'batch_size': 256,
        'num_epochs': 200,
        'learning_rate': 0.01,
        'weight_decay': 5e-4,
        'selection_interval': 50,
        'clc_beta': 1e5,
        'expected_top1': 71.25,
        'expected_top5': 91.39,
    }


def create_config_from_experiment(experiment_config):
    """Create a config module from experiment configuration"""
    import config
    
    # Update config values
    for key, value in experiment_config.items():
        if hasattr(config, key.upper()):
            setattr(config, key.upper(), value)
    
    return config


def run_experiment(experiment_config, device='cuda'):
    """Run a single experiment"""
    
    logger = logging.getLogger(__name__)
    logger.info("\n" + "="*70)
    logger.info(f"Running Experiment: {experiment_config['name']}")
    logger.info("="*70)
    
    # Log experiment details
    logger.info(f"Model: {experiment_config['model']}")
    logger.info(f"Dataset: {experiment_config['dataset']}")
    logger.info(f"Quantization: {experiment_config['weight_bits']}w{experiment_config['activation_bits']}a")
    logger.info(f"Coreset Fraction: {experiment_config['coreset_fraction']*100:.1f}%")
    logger.info(f"Expected Top-1: {experiment_config['expected_top1']:.2f}%")
    logger.info(f"Expected Top-5: {experiment_config['expected_top5']:.2f}%")
    logger.info("="*70)
    
    # Import and update config
    import importlib
    import config
    importlib.reload(config)
    
    # Update config with experiment settings
    config.DATASET = experiment_config['dataset']
    config.MODEL_NAME = experiment_config['model']
    config.WEIGHT_BITS = experiment_config['weight_bits']
    config.ACTIVATION_BITS = experiment_config['activation_bits']
    config.CORESET_FRACTION = experiment_config['coreset_fraction']
    config.BATCH_SIZE = experiment_config['batch_size']
    config.NUM_EPOCHS = experiment_config['num_epochs']
    config.LEARNING_RATE = experiment_config['learning_rate']
    config.WEIGHT_DECAY = experiment_config['weight_decay']
    config.SELECTION_INTERVAL = experiment_config['selection_interval']
    config.CLC_BETA = experiment_config['clc_beta']
    config.DEVICE = device
    
    # Run training
    from main import main as train_main
    
    try:
        top1_acc, top5_acc = train_main()
        
        # Compare with expected results
        top1_diff = top1_acc - experiment_config['expected_top1']
        top5_diff = top5_acc - experiment_config['expected_top5']
        
        logger.info("\n" + "="*70)
        logger.info("Results Summary:")
        logger.info("="*70)
        logger.info(f"Achieved Top-1: {top1_acc:.2f}% (Expected: {experiment_config['expected_top1']:.2f}%, Diff: {top1_diff:+.2f}%)")
        logger.info(f"Achieved Top-5: {top5_acc:.2f}% (Expected: {experiment_config['expected_top5']:.2f}%, Diff: {top5_diff:+.2f}%)")
        logger.info("="*70)
        
        return {
            'name': experiment_config['name'],
            'top1_achieved': top1_acc,
            'top1_expected': experiment_config['expected_top1'],
            'top5_achieved': top5_acc,
            'top5_expected': experiment_config['expected_top5'],
        }
    
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        return None


def main():
    """Run multiple experiments"""
    
    # Setup logging
    os.makedirs('./logs', exist_ok=True)
    log_file = os.path.join('./logs', f'experiments_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info("\n" + "="*70)
    logger.info("QuaRC Experiment Runner")
    logger.info("="*70)
    logger.info(f"Device: {device}")
    logger.info("="*70)
    
    # Run experiments
    experiments = [
        ExperimentConfig.CIFAR100_MOBILENETV2_2BIT_1PERCENT,
        ExperimentConfig.CIFAR100_MOBILENETV2_3BIT_1PERCENT,
        ExperimentConfig.CIFAR100_MOBILENETV2_4BIT_1PERCENT,
        ExperimentConfig.CIFAR100_MOBILENETV2_2BIT_5PERCENT,
        ExperimentConfig.CIFAR100_MOBILENETV2_3BIT_5PERCENT,
    ]
    
    results = []
    
    for experiment_config in experiments:
        result = run_experiment(experiment_config, device=device)
        if result:
            results.append(result)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY - All Experiments")
    logger.info("="*70)
    
    for result in results:
        logger.info(f"\n{result['name']}:")
        logger.info(f"  Top-1: {result['top1_achieved']:.2f}% (Expected: {result['top1_expected']:.2f}%)")
        logger.info(f"  Top-5: {result['top5_achieved']:.2f}% (Expected: {result['top5_expected']:.2f}%)")
    
    logger.info("="*70)


if __name__ == '__main__':
    main()
