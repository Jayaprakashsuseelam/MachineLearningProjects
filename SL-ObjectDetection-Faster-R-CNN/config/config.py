"""
Main configuration file for SL-ObjectDetection-Faster-R-CNN
"""
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class DataConfig:
    """Data configuration settings"""
    # Dataset paths
    data_root: str = "./data"
    voc_root: str = "./data/VOCdevkit"
    voc_year: str = "2012"
    
    # Data processing
    image_size: Tuple[int, int] = (800, 800)
    min_size: int = 600
    max_size: int = 1000
    
    # Augmentation
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    color_jitter_prob: float = 0.3
    random_crop_prob: float = 0.3
    
    # Batch settings
    batch_size: int = 2
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class ModelConfig:
    """Model configuration settings"""
    # Backbone
    backbone: str = "resnet50"
    pretrained: bool = True
    freeze_backbone: bool = False
    
    # RPN settings
    anchor_scales: List[int] = None
    anchor_ratios: List[float] = None
    rpn_pre_nms_top_n_train: int = 2000
    rpn_post_nms_top_n_train: int = 2000
    rpn_pre_nms_top_n_test: int = 1000
    rpn_post_nms_top_n_test: int = 1000
    rpn_nms_thresh: float = 0.7
    rpn_fg_iou_thresh: float = 0.7
    rpn_bg_iou_thresh: float = 0.3
    rpn_batch_size_per_image: int = 256
    rpn_positive_fraction: float = 0.5
    
    # RoI settings
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    bbox_reg_weights: Optional[List[float]] = None
    
    # Detection head
    num_classes: int = 21  # PASCAL VOC: 20 + background
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100


@dataclass
class TrainingConfig:
    """Training configuration settings"""
    # General training
    epochs: int = 12
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    
    # Learning rate scheduling
    lr_scheduler: str = "step"  # "step", "cosine", "warmup_cosine"
    lr_step_size: int = 3
    lr_gamma: float = 0.1
    warmup_epochs: int = 1
    warmup_factor: float = 0.1
    
    # Optimization
    optimizer: str = "sgd"  # "sgd", "adam", "adamw"
    gradient_clip: float = 5.0
    mixed_precision: bool = False
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_freq: int = 1
    resume_from: Optional[str] = None
    
    # Logging
    log_dir: str = "./logs"
    log_freq: int = 100
    tensorboard: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation configuration settings"""
    # Metrics
    iou_thresholds: List[float] = None
    max_det: int = 100
    
    # Visualization
    save_predictions: bool = True
    save_dir: str = "./results/predictions"
    show_images: bool = False
    confidence_threshold: float = 0.5


@dataclass
class Config:
    """Main configuration class"""
    # Environment
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    seed: int = 42
    deterministic: bool = True
    
    # Sub-configs
    data: DataConfig = None
    model: ModelConfig = None
    training: TrainingConfig = None
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        """Initialize default sub-configs if not provided"""
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.training is None:
            self.training = TrainingConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()
        
        # Set default anchor settings
        if self.model.anchor_scales is None:
            self.model.anchor_scales = [8, 16, 32]
        if self.model.anchor_ratios is None:
            self.model.anchor_ratios = [0.5, 1.0, 2.0]
        if self.model.bbox_reg_weights is None:
            self.model.bbox_reg_weights = [1.0, 1.0, 1.0, 1.0]
        if self.evaluation.iou_thresholds is None:
            self.evaluation.iou_thresholds = [0.5, 0.75]
    
    def save(self, path: str):
        """Save configuration to file"""
        import yaml
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Convert dataclass to dict
        config_dict = {
            'device': self.device,
            'seed': self.seed,
            'deterministic': self.deterministic,
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file"""
        import yaml
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Reconstruct dataclass objects
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        
        return cls(
            device=config_dict.get('device', 'cuda'),
            seed=config_dict.get('seed', 42),
            deterministic=config_dict.get('deterministic', True),
            data=data_config,
            model=model_config,
            training=training_config,
            evaluation=evaluation_config
        )


# Default configuration
default_config = Config()


def get_config(config_path: Optional[str] = None) -> Config:
    """Get configuration object"""
    if config_path and os.path.exists(config_path):
        return Config.load(config_path)
    return default_config
