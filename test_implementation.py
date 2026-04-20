"""
Test script to verify QuaRC implementation
"""
import sys
import torch
import torch.nn as nn

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config module imported")
    except Exception as e:
        print(f"✗ Failed to import config: {e}")
        return False
    
    try:
        from data_loader import get_cifar100_loaders, get_full_dataset
        print("✓ data_loader module imported")
    except Exception as e:
        print(f"✗ Failed to import data_loader: {e}")
        return False
    
    try:
        from model_utils import get_model, quantize_model, set_seed
        print("✓ model_utils module imported")
    except Exception as e:
        print(f"✗ Failed to import model_utils: {e}")
        return False
    
    try:
        from quantization import FakeQuantization, QuantizationModule
        print("✓ quantization module imported")
    except Exception as e:
        print(f"✗ Failed to import quantization: {e}")
        return False
    
    try:
        from coreset_selection import calculate_relative_entropy, select_coreset
        print("✓ coreset_selection module imported")
    except Exception as e:
        print(f"✗ Failed to import coreset_selection: {e}")
        return False
    
    try:
        from trainer import QuantizationAwareTrainer, KnowledgeDistillationLoss
        print("✓ trainer module imported")
    except Exception as e:
        print(f"✗ Failed to import trainer: {e}")
        return False
    
    return True


def test_fake_quantization():
    """Test fake quantization operations"""
    print("\nTesting fake quantization...")
    
    try:
        from quantization import FakeQuantization
        
        # Create a random tensor
        x = torch.randn(4, 10)
        
        # Create fake quantizer
        quantizer = FakeQuantization(bits=2, symmetric=True)
        quantizer.initialize_scale(x)
        
        # Quantize
        x_q = quantizer(x)
        
        print(f"✓ Fake quantization works")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {x_q.shape}")
        print(f"  Scale: {quantizer.scale.item():.6f}")
        
        return True
    except Exception as e:
        print(f"✗ Fake quantization test failed: {e}")
        return False


def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    try:
        from model_utils import get_model
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        model = get_model('mobilenetv2', num_classes=100, pretrained=False, device=device)
        
        print(f"✓ Model loaded successfully")
        print(f"  Model: MobileNetV2")
        print(f"  Device: {device}")
        print(f"  Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 32, 32).to(device)
        y = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {y.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Model loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_coreset_selection():
    """Test coreset selection functions"""
    print("\nTesting coreset selection...")
    
    try:
        from coreset_selection import calculate_relative_entropy, select_coreset
        import torch
        
        # Create random logits
        p_q = torch.randn(8, 10)  # Quantized model outputs
        p_f = torch.randn(8, 10)  # Full-precision model outputs
        
        # Calculate RES
        res = calculate_relative_entropy(p_q, p_f, reduction='none')
        
        print(f"✓ RES calculation works")
        print(f"  Batch size: {p_q.shape[0]}")
        print(f"  RES shape: {res.shape}")
        print(f"  RES values: min={res.min():.4f}, max={res.max():.4f}, mean={res.mean():.4f}")
        
        # Test coreset selection
        scores = res.numpy()
        indices = np.arange(len(scores))
        coreset_indices = select_coreset(scores, indices, coreset_fraction=0.5)
        
        print(f"✓ Coreset selection works")
        print(f"  Total samples: {len(indices)}")
        print(f"  Selected samples: {len(coreset_indices)} (50%)")
        
        return True
    except Exception as e:
        print(f"✗ Coreset selection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer():
    """Test trainer initialization"""
    print("\nTesting trainer...")
    
    try:
        from trainer import QuantizationAwareTrainer, KnowledgeDistillationLoss
        from model_utils import get_model
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load models
        fp_model = get_model('mobilenetv2', num_classes=100, pretrained=False, device=device)
        q_model = get_model('mobilenetv2', num_classes=100, pretrained=False, device=device)
        
        # Setup optimizer
        optimizer = torch.optim.SGD(q_model.parameters(), lr=0.01)
        
        # Create trainer
        trainer = QuantizationAwareTrainer(
            fp_model=fp_model,
            q_model=q_model,
            optimizer=optimizer,
            device=device,
            use_kd=True,
            use_clc=False,  # CLC is optional for this test
        )
        
        print(f"✓ Trainer initialized successfully")
        print(f"  Device: {device}")
        print(f"  KD Loss: Enabled")
        print(f"  CLC Loss: Disabled (for testing)")
        
        return True
    except Exception as e:
        print(f"✗ Trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_device():
    """Test device configuration"""
    print("Testing device configuration...")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*50)
    print("QuaRC Implementation Tests")
    print("="*50 + "\n")
    
    # Import numpy for coreset test
    global np
    import numpy as np
    
    tests = [
        ("Device Configuration", test_device),
        ("Module Imports", test_imports),
        ("Fake Quantization", test_fake_quantization),
        ("Model Loading", test_model_loading),
        ("Coreset Selection", test_coreset_selection),
        ("Trainer", test_trainer),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("Test Summary")
    print("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print("="*50)
    print(f"Total: {passed}/{total} tests passed")
    print("="*50 + "\n")
    
    if passed == total:
        print("✓ All tests passed! Ready to run experiments.")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
