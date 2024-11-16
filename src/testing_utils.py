import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging
from model import NLIModelWithDebias
from data_handler import get_dataloaders

class ModelTester:
    def __init__(self, model_path: str, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.logger = logging.getLogger('model_tester')
    def load_model(self, checkpoint_path: str) -> NLIModelWithDebias:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model = NLIModelWithDebias()
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        return model
    def test_model_outputs(self, model: NLIModelWithDebias, batch: Dict) -> Dict:
        with torch.no_grad():
            outputs = model(**{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        return {
            'loss': outputs['loss'].item() if outputs['loss'] is not None else None,
            'logits': outputs['logits'].cpu().numpy(),
            'predictions': outputs['logits'].argmax(dim=-1).cpu().numpy(),
            'embeddings': outputs['cls_embedding'].cpu().numpy()
        }
    def verify_checkpoints(self) -> List[Dict]:
        checkpoint_info = []
        for checkpoint_dir in sorted(self.model_path.glob('checkpoint-*')):
            checkpoint_path = checkpoint_dir / 'model.pt'
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    info = {
                        'checkpoint': checkpoint_dir.name,
                        'global_step': checkpoint.get('global_step', 0),
                        'best_accuracy': checkpoint.get('best_accuracy', 0.0),
                        'size_mb': checkpoint_path.stat().st_size / (1024 * 1024)
                    }
                    checkpoint_info.append(info)
                except Exception as e:
                    self.logger.warning(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return checkpoint_info
    def validate_training_progress(self) -> Dict:
        metrics_files = sorted(self.model_path.glob('epoch_*_metrics.json'))
        metrics_history = []
        for file in metrics_files:
            with open(file) as f:
                metrics = json.load(f)
                metrics_history.append(metrics)
        if not metrics_history:
            raise ValueError("No training metrics found")
        validation = {
            'loss_decreasing': self._check_loss_trend(metrics_history),
            'accuracy_improving': self._check_accuracy_trend(metrics_history),
            'bias_metrics_improving': self._check_bias_metrics(metrics_history),
            'final_metrics': metrics_history[-1]['validation']
        }
        return validation
    def _check_loss_trend(self, metrics: List[Dict]) -> bool:
        losses = [m['train']['loss'] for m in metrics]
        return np.mean([losses[i] > losses[i+1] for i in range(len(losses)-1)]) > 0.6
    def _check_accuracy_trend(self, metrics: List[Dict]) -> bool:
        accuracies = [m['validation']['accuracy'] for m in metrics]
        return np.mean([accuracies[i] < accuracies[i+1] for i in range(len(accuracies)-1)]) > 0.5
    def _check_bias_metrics(self, metrics: List[Dict]) -> Dict:
        initial_metrics = metrics[0]['validation']
        final_metrics = metrics[-1]['validation']
        return {
            'length_bias_improved': (final_metrics.get('length_bias_accuracy', 0) > initial_metrics.get('length_bias_accuracy', 0)),
            'overlap_bias_improved': (final_metrics.get('overlap_bias_accuracy', 0) > initial_metrics.get('overlap_bias_accuracy', 0))
        }

class PerformanceProfiler:
    def __init__(self, model: NLIModelWithDebias, device: torch.device):
        self.model = model
        self.device = device
    def profile_memory_usage(self, batch: Dict) -> Dict:
        torch.cuda.reset_peak_memory_stats()
        outputs = self.model(**{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**2,
            'peak_allocated': torch.cuda.max_memory_allocated() / 1024**2,
            'cached': torch.cuda.memory_reserved() / 1024**2
        }
    def profile_inference_time(self, batch: Dict, num_runs: int = 100) -> Dict:
        timings = []
        for _ in range(10):
            _ = self.model(**{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        for _ in range(num_runs):
            starter.record()
            _ = self.model(**{k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()})
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))
        return {
            'mean_time_ms': np.mean(timings),
            'std_time_ms': np.std(timings),
            'min_time_ms': np.min(timings),
            'max_time_ms': np.max(timings)
        }
def run_model_tests():
    tester = ModelTester('debiased_model')
    logger = logging.getLogger('model_tests')
    try:
        logger.info("Verifying checkpoints...")
        checkpoint_info = tester.verify_checkpoints()
        logger.info("Validating training progress...")
        validation = tester.validate_training_progress()
        logger.info("Testing final model...")
        model = tester.load_model('debiased_model/best_model/model.pt')
        _, val_loader = get_dataloaders(batch_size=16)
        test_batch = next(iter(val_loader))
        outputs = tester.test_model_outputs(model, test_batch)
        profiler = PerformanceProfiler(model, tester.device)
        memory_usage = profiler.profile_memory_usage(test_batch)
        timing_stats = profiler.profile_inference_time(test_batch)
        test_results = {
            'checkpoint_info': checkpoint_info,
            'training_validation': validation,
            'memory_usage': memory_usage,
            'timing_stats': timing_stats
        }
        return test_results
    except Exception as e:
        logger.error(f"Model testing failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_model_tests()
    print(json.dumps(results, indent=2))
