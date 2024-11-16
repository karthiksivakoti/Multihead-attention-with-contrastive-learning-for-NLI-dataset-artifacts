import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def prepare_output_directories(base_dir: str = "results") -> Dict[str, Path]:
    base_path = Path(base_dir)
    directories = {
        'base': base_path,
        'checkpoints': base_path / 'checkpoints',
        'metrics': base_path / 'metrics',
        'figures': base_path / 'figures',
        'analysis': base_path / 'analysis'
    }
    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    return directories
def compute_accuracy_by_category(predictions: List[int], labels: List[int], categories: Dict[str, List[bool]]) -> Dict[str, float]:
    metrics = {}
    for category_name, category_mask in categories.items():
        if sum(category_mask) > 0:
            category_preds = np.array(predictions)[category_mask]
            category_labels = np.array(labels)[category_mask]
            metrics[f'{category_name}_accuracy'] = np.mean(category_preds == category_labels)
            metrics[f'{category_name}_count'] = sum(category_mask)
    return metrics
def analyze_confidence_distribution(logits: torch.Tensor, predictions: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    probabilities = torch.softmax(logits, dim=-1)
    confidences = probabilities.max(dim=-1).values
    correct_mask = predictions == labels
    incorrect_mask = ~correct_mask
    metrics = {
        'mean_confidence': confidences.mean().item(),
        'mean_confidence_correct': confidences[correct_mask].mean().item(),
        'mean_confidence_incorrect': confidences[incorrect_mask].mean().item() if incorrect_mask.any() else 0.0,
        'high_confidence_rate': (confidences > 0.9).float().mean().item()
    }
    return metrics
def save_experiment_config(config: Dict[str, Any], save_path: Path):
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
def load_experiment_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return json.load(f)
def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    return logger
