#!/usr/bin/env python3
"""
Quick Start Example for Faster R-CNN
"""
import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models import faster_rcnn_resnet50
from data import VOCDataset, get_transform
from utils import calculate_map


def main():
    """Quick start example"""
    print("🚀 Faster R-CNN Quick Start Example")
    print("=" * 50)
    
    # 1. Create model
    print("\n1. Creating Faster R-CNN model...")
    model = faster_rcnn_resnet50(num_classes=21, pretrained=True)
    print(f"   ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 2. Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"   ✓ Model moved to {device}")
    
    # 3. Create dataset
    print("\n2. Loading PASCAL VOC dataset...")
    try:
        dataset = VOCDataset(root='../data/VOCdevkit', year='2012', image_set='train')
        print(f"   ✓ Dataset loaded with {len(dataset)} samples")
    except Exception as e:
        print(f"   ⚠ Dataset not found: {e}")
        print("   Please download PASCAL VOC dataset to ../data/VOCdevkit")
        return
    
    # 4. Get transforms
    print("\n3. Setting up data transforms...")
    transform = get_transform(train=True, image_size=(800, 800))
    print("   ✓ Transforms configured")
    
    # 5. Load a sample
    print("\n4. Loading sample data...")
    image, target = dataset[0]
    print(f"   ✓ Sample loaded - Image shape: {image.shape if hasattr(image, 'shape') else 'PIL Image'}")
    print(f"   ✓ Target contains {len(target['boxes'])} objects")
    
    # 6. Apply transforms
    print("\n5. Applying transforms...")
    try:
        transformed_image, transformed_target = transform(image, target)
        print(f"   ✓ Transforms applied - Image shape: {transformed_image.shape}")
    except Exception as e:
        print(f"   ⚠ Transform failed: {e}")
        return
    
    # 7. Model inference
    print("\n6. Running model inference...")
    model.eval()
    with torch.no_grad():
        # Prepare input
        if isinstance(transformed_image, torch.Tensor):
            input_tensor = transformed_image.unsqueeze(0).to(device)
        else:
            input_tensor = torch.from_numpy(np.array(transformed_image)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        # Forward pass
        outputs = model(input_tensor)
        print("   ✓ Inference completed")
        
        # Check outputs
        if 'detections' in outputs:
            detections = outputs['detections'][0]
            print(f"   ✓ Found {len(detections['boxes'])} detections")
        else:
            print("   ⚠ No detections in output")
    
    # 8. Summary
    print("\n" + "=" * 50)
    print("✅ Quick Start Example Completed Successfully!")
    print("\nNext steps:")
    print("1. Train the model: python scripts/train.py")
    print("2. Run inference: python scripts/demo.py --checkpoint path/to/checkpoint --image path/to/image")
    print("3. Explore notebooks: jupyter notebook notebooks/")
    print("\nHappy coding! 🎉")


if __name__ == '__main__':
    main()
