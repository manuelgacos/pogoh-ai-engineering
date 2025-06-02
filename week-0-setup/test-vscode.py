#!/usr/bin/env python3
"""
VS Code AI Development Environment Test
"""

import pandas as pd
import numpy as np
import torch
import transformers
from datetime import datetime

def test_environment():
    """Test all major components work in VS Code"""
    
    print("🧪 Testing VS Code AI Development Environment")
    print("=" * 50)
    
    # Test data manipulation
    df = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'value': np.random.randn(10)
    })
    print(f"✓ Created test DataFrame: {df.shape}")
    
    # Test PyTorch
    tensor = torch.randn(3, 4)
    print(f"✓ Created PyTorch tensor: {tensor.shape}")
    
    # Test if GPU available
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    # Test transformers library
    print(f"✓ Transformers version: {transformers.__version__}")
    
    print("=" * 50)
    print("🎉 VS Code environment ready for AI development!")

if __name__ == "__main__":
    test_environment()