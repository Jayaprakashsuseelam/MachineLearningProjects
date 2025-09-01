# SL-ObjectDetection-Faster-R-CNN Project Structure

This document provides a comprehensive overview of the project structure, explaining the purpose and organization of each component in the Faster R-CNN implementation.

## 📁 Root Directory

```
SL-ObjectDetection-Faster-R-CNN/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation configuration
├── PROJECT_STRUCTURE.md         # This file - project structure overview
├── config/                      # Configuration management
├── src/                         # Source code
├── scripts/                     # Executable scripts
├── examples/                    # Example usage and tutorials
├── tests/                       # Unit tests
└── data/                        # Data storage (created during runtime)
```

## 🗂️ Configuration (`config/`)

The configuration system provides centralized management of all project parameters.

```
config/
├── __init__.py                 # Package initialization
├── config.py                   # Main configuration classes and utilities
└── default_config.yaml         # Default configuration file
```

### Key Components:
- **`Config`**: Main configuration dataclass that aggregates all settings
- **`DataConfig`**: Dataset and data loading parameters
- **`ModelConfig`**: Model architecture and hyperparameters
- **`TrainingConfig`**: Training loop and optimization settings
- **`EvaluationConfig`**: Evaluation metrics and thresholds

### Features:
- YAML/JSON configuration file support
- Environment variable overrides
- Configuration validation and defaults
- Easy saving/loading of configurations

## 🔧 Source Code (`src/`)

The core implementation is organized into logical modules for maintainability and reusability.

```
src/
├── __init__.py                 # Package initialization and top-level imports
├── models/                     # Model implementations
├── data/                       # Data handling components
├── training/                   # Training infrastructure
└── utils/                      # Utility functions
```

### 🧠 Models (`src/models/`)

Contains all neural network architectures and components.

```
models/
├── __init__.py                 # Model package initialization
├── backbone.py                 # Feature extraction networks
├── rpn.py                      # Region Proposal Network
├── roi_pooling.py             # Region of Interest pooling layers
├── faster_rcnn.py             # Main Faster R-CNN model
└── losses.py                   # Loss functions
```

#### Key Components:

**`backbone.py`**
- **`Backbone`**: Abstract base class for feature extractors
- **`ResNetBackbone`**: ResNet-based backbone implementation
- **`VGGBackbone`**: VGG-based backbone implementation
- **`get_backbone()`**: Factory function for creating backbones

**`rpn.py`**
- **`AnchorGenerator`**: Generates anchor boxes for RPN
- **`RPNHead`**: RPN classification and regression heads
- **`RPN`**: Complete Region Proposal Network implementation

**`roi_pooling.py`**
- **`RoIPooling`**: Standard RoI pooling implementation
- **`RoIAlign`**: More accurate RoI alignment
- **`RoIPoolingV2`**: Improved RoI pooling with better gradients
- **`RoIPooling3D`**: 3D variant for volumetric data

**`faster_rcnn.py`**
- **`FasterRCNN`**: Main model integrating all components
- **`FastRCNNPredictor`**: Classification and regression heads
- **`faster_rcnn_resnet50/101/vgg16()`**: Pre-configured model variants

**`losses.py`**
- **`RPNLoss`**: RPN training loss (classification + regression)
- **`DetectionLoss`**: Detection head training loss
- **`CombinedLoss`**: Aggregates RPN and detection losses
- **`FocalLoss`**: Handles class imbalance
- **`IoULoss`**: Bounding box regression loss

### 📊 Data (`src/data/`)

Handles dataset loading, preprocessing, and augmentation.

```
data/
├── __init__.py                 # Data package initialization
├── dataset.py                  # Dataset implementations
├── transforms.py               # Data augmentation and preprocessing
└── collate.py                  # Custom collate functions
```

#### Key Components:

**`dataset.py`**
- **`VOCDataset`**: PASCAL VOC dataset implementation
- **`CustomDataset`**: Base class for custom datasets
- **`COCODataset`**: COCO dataset implementation
- **`get_dataset()`**: Factory function for dataset creation

**`transforms.py`**
- **`Compose`**: Chains multiple transformations
- **`ToTensor`**: Converts PIL images to tensors
- **`Normalize`**: Normalizes image values
- **`Resize`**: Resizes images and bounding boxes
- **`RandomHorizontalFlip`**: Random horizontal flipping
- **`ColorJitter`**: Color augmentation
- **`RandomCrop`**: Random cropping
- **`RandomRotation`**: Random rotation
- **`RandomErasing`**: Random erasing augmentation
- **`PadToSize`**: Pads images to fixed size

**`collate.py`**
- **`collate_fn()`**: Main collate function for variable-sized inputs
- **`collate_fn_simple()`**: Simplified version for basic use cases
- **`collate_fn_with_metadata()`**: Preserves all metadata

### 🚀 Training (`src/training/`)

Provides the training infrastructure and optimization utilities.

```
training/
├── __init__.py                 # Training package initialization
├── trainer.py                  # Main training loop
└── optimizer.py                # Optimizer and scheduler factories
```

#### Key Components:

**`trainer.py`**
- **`Trainer`**: Complete training orchestration class
- **`create_trainer()`**: Factory function for trainer creation

**`optimizer.py`**
- **`get_optimizer()`**: Creates optimizers (SGD, Adam, AdamW, etc.)
- **`get_scheduler()`**: Creates learning rate schedulers
- **`get_optimizer_with_groups()`**: Creates optimizers with parameter groups
- **`create_optimizer_groups()`**: Groups parameters by model component

### 🛠️ Utils (`src/utils/`)

Provides utility functions for visualization, metrics, and general helpers.

```
utils/
├── __init__.py                 # Utils package initialization
├── visualization.py            # Visualization utilities
├── metrics.py                  # Evaluation metrics
└── helpers.py                  # Helper functions
```

#### Key Components:

**`visualization.py`**
- **`visualize_predictions()`**: Visualizes model predictions
- **`plot_detections()`**: Plots detections with bounding boxes
- **`create_detection_video()`**: Creates video with detection overlays
- **`create_detection_grid()`**: Grid visualization of multiple images
- **`save_detection_image()`**: Saves images with detections

**`metrics.py`**
- **`calculate_map()`**: Mean Average Precision calculation
- **`calculate_iou()`**: Intersection over Union calculation
- **`calculate_precision_recall()`**: Precision-recall curves
- **`plot_precision_recall_curve()`**: Plots PR curves
- **`calculate_class_wise_metrics()`**: Per-class performance metrics

**`helpers.py`**
- **`get_device_info()`**: Hardware information and capabilities
- **`save_checkpoint()`** / **`load_checkpoint()`**: Model checkpointing
- **`save_config()`** / **`load_config()`**: Configuration management
- **`count_parameters()`**: Model parameter counting
- **`set_seed()`**: Reproducibility utilities
- **`create_experiment_dir()`**: Experiment organization

## 📜 Scripts (`scripts/`)

Executable scripts for common tasks.

```
scripts/
├── train.py                    # Training script
└── demo.py                     # Inference demonstration script
```

### Key Scripts:

**`train.py`**
- Command-line training interface
- Configuration file loading
- Model and dataset initialization
- Training execution and evaluation

**`demo.py`**
- Single image inference
- Model loading from checkpoints
- Prediction visualization
- Result saving

## 📚 Examples (`examples/`)

Example usage and tutorials.

```
examples/
├── quick_start.py              # Quick start guide
└── faster_rcnn_case_study.py   # Comprehensive case study
```

### Key Examples:

**`quick_start.py`**
- Basic model creation and usage
- Simple inference demonstration
- Dataset loading example

**`faster_rcnn_case_study.py`**
- Complete tutorial implementation
- Theoretical background explanation
- Practical implementation steps
- Performance analysis

## 🧪 Tests (`tests/`)

Unit tests for code validation.

```
tests/
├── test_basic_imports.py      # Basic functionality tests
└── ...                        # Additional test files
```

### Test Coverage:
- Module import validation
- Model creation and instantiation
- Configuration loading
- Basic functionality verification

## 📁 Data Directory (`data/`)

Created during runtime for dataset storage.

```
data/
├── VOC/                        # PASCAL VOC dataset
│   ├── VOCdevkit/
│   └── VOC2012/
└── checkpoints/                # Model checkpoints
```

## 🔄 Workflow

### 1. **Setup and Configuration**
```python
from config import get_config
config = get_config("config/default_config.yaml")
```

### 2. **Model Creation**
```python
from models import faster_rcnn_resnet50
model = faster_rcnn_resnet50(num_classes=21, pretrained=True)
```

### 3. **Data Loading**
```python
from data import VOCDataset, get_transform
dataset = VOCDataset(root="./data/VOC", transforms=get_transform(train=True))
```

### 4. **Training**
```python
from training import Trainer
trainer = Trainer(model, dataset, config)
trainer.train()
```

### 5. **Evaluation**
```python
from utils import calculate_map
mAP = calculate_map(predictions, targets)
```

### 6. **Inference**
```python
from utils import visualize_predictions
visualize_predictions(image, predictions, class_names)
```

## 🎯 Key Design Principles

### 1. **Modularity**
- Each component is self-contained and focused
- Clear interfaces between modules
- Easy to extend and modify

### 2. **Configurability**
- Centralized configuration management
- Environment-specific settings
- Easy hyperparameter tuning

### 3. **Reusability**
- Factory functions for common operations
- Base classes for extensibility
- Utility functions for common tasks

### 4. **Maintainability**
- Clear separation of concerns
- Comprehensive documentation
- Consistent coding style

### 5. **Performance**
- Efficient data loading with custom collate functions
- Optimized model architectures
- GPU acceleration support

## 🚀 Getting Started

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Setup configuration**: Modify `config/default_config.yaml`
3. **Run quick start**: `python examples/quick_start.py`
4. **Run case study**: `python examples/faster_rcnn_case_study.py`
5. **Train model**: `python scripts/train.py`
6. **Run inference**: `python scripts/demo.py`

## 🔧 Customization

### Adding New Backbones
```python
# In src/models/backbone.py
class CustomBackbone(Backbone):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your implementation
```

### Adding New Datasets
```python
# In src/data/dataset.py
class CustomDataset(CustomDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Your implementation
```

### Adding New Loss Functions
```python
# In src/models/losses.py
class CustomLoss(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # Your implementation
```

This project structure provides a solid foundation for object detection research and development, with clear organization and extensibility for future enhancements.
