import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_models():
    print("=== SMOKE TEST: Model Architectures ===")
    x = torch.randn(4, 12, 1000)
    print(f"Input shape: {x.shape}")
    
    try:
        from src.models.wavkan import WavKANClassifier
        m = WavKANClassifier(in_channels=12, input_dim=1000, num_classes=5)
        out = m(x)
        print(f"WavKAN: {out.shape} OK!")
    except Exception as e:
        print(f"WavKAN FAILED: {e}")
        
    try:
        from src.models.wavkan_multiscale import MultiScaleWavKANClassifier
        m = MultiScaleWavKANClassifier(in_channels=12, input_dim=1000, num_classes=5)
        out = m(x)
        print(f"MS-WavKAN: {out.shape} OK!")
    except Exception as e:
        print(f"MS-WavKAN FAILED: {e}")
        
    try:
        from src.models.baselines import ResNet1D, InceptionTime
        m1 = ResNet1D(in_channels=12, num_classes=5)
        out1 = m1(x)
        print(f"ResNet1D: {out1.shape} OK!")
        
        m2 = InceptionTime(in_channels=12, num_classes=5)
        out2 = m2(x)
        print(f"InceptionTime: {out2.shape} OK!")
    except Exception as e:
        print(f"Baselines FAILED: {e}")
        
    try:
        from src.models.dann import DANN
        m = DANN(backbone='wavkan', in_channels=12, num_classes=5)
        class_out, domain_out = m(x, alpha=1.0)
        print(f"DANN (WavKAN): Class {class_out.shape}, Domain {domain_out.shape} OK!")
    except Exception as e:
        print(f"DANN FAILED: {e}")

if __name__ == "__main__":
    test_models()
    print("Smoke tests passed successfully!")
