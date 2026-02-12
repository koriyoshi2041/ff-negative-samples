"""
Comprehensive Logging Utilities for FF Experiments
===================================================
Ensures all data is properly recorded for later visualization.
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import torch
import numpy as np


class ExperimentLogger:
    """
    Comprehensive experiment logger that saves:
    - Configuration
    - Training history (loss, accuracy per epoch)
    - Layer-by-layer metrics
    - Timing information
    - Final results
    """

    def __init__(self, experiment_name: str, output_dir: str = "results"):
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = time.time()
        self.timestamp = datetime.now().isoformat()

        self.data = {
            "experiment_name": experiment_name,
            "timestamp": self.timestamp,
            "config": {},
            "training_history": {},
            "layer_metrics": {},
            "results": {},
            "timing": {},
            "device_info": {}
        }

        # Detect device
        if torch.cuda.is_available():
            self.data["device_info"] = {
                "type": "cuda",
                "name": torch.cuda.get_device_name(0),
                "count": torch.cuda.device_count()
            }
        elif torch.backends.mps.is_available():
            self.data["device_info"] = {"type": "mps"}
        else:
            self.data["device_info"] = {"type": "cpu"}

    def log_config(self, config: Dict[str, Any]):
        """Log experiment configuration."""
        self.data["config"] = self._convert_to_serializable(config)

    def log_epoch(self, model_name: str, layer_idx: int, epoch: int,
                  loss: float, pos_goodness: float = None, neg_goodness: float = None,
                  accuracy: float = None, extra: Dict = None):
        """Log metrics for a single epoch."""
        key = f"{model_name}_layer{layer_idx}"
        if key not in self.data["training_history"]:
            self.data["training_history"][key] = {
                "epochs": [],
                "loss": [],
                "pos_goodness": [],
                "neg_goodness": [],
                "accuracy": []
            }

        history = self.data["training_history"][key]
        history["epochs"].append(epoch)
        history["loss"].append(float(loss))
        if pos_goodness is not None:
            history["pos_goodness"].append(float(pos_goodness))
        if neg_goodness is not None:
            history["neg_goodness"].append(float(neg_goodness))
        if accuracy is not None:
            history["accuracy"].append(float(accuracy))

        if extra:
            for k, v in extra.items():
                if k not in history:
                    history[k] = []
                history[k].append(self._convert_to_serializable(v))

    def log_layer_complete(self, model_name: str, layer_idx: int,
                           final_loss: float, training_time: float,
                           final_accuracy: float = None):
        """Log completion of a layer's training."""
        key = f"{model_name}_layer{layer_idx}"
        self.data["layer_metrics"][key] = {
            "final_loss": float(final_loss),
            "training_time_seconds": float(training_time),
            "final_accuracy": float(final_accuracy) if final_accuracy else None
        }

    def log_model_complete(self, model_name: str, train_acc: float, test_acc: float,
                           total_time: float, extra: Dict = None):
        """Log completion of model training."""
        self.data["results"][model_name] = {
            "train_accuracy": float(train_acc),
            "test_accuracy": float(test_acc),
            "total_training_time_seconds": float(total_time)
        }
        if extra:
            self.data["results"][model_name].update(
                self._convert_to_serializable(extra)
            )

    def log_transfer_result(self, model_name: str, source_acc: float,
                            transfer_acc: float, transfer_history: List[float] = None,
                            extra: Dict = None):
        """Log transfer learning results."""
        if "transfer" not in self.data["results"]:
            self.data["results"]["transfer"] = {}

        self.data["results"]["transfer"][model_name] = {
            "source_accuracy": float(source_acc),
            "transfer_accuracy": float(transfer_acc),
            "transfer_history": transfer_history if transfer_history else []
        }
        if extra:
            self.data["results"]["transfer"][model_name].update(
                self._convert_to_serializable(extra)
            )

    def log_final_summary(self, summary: Dict[str, Any]):
        """Log final summary and analysis."""
        self.data["summary"] = self._convert_to_serializable(summary)

    def save(self, filename: str = None):
        """Save all logged data to JSON."""
        self.data["timing"]["total_experiment_seconds"] = time.time() - self.start_time
        self.data["timing"]["end_timestamp"] = datetime.now().isoformat()

        os.makedirs(self.output_dir, exist_ok=True)

        if filename is None:
            filename = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        filepath = os.path.join(self.output_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=2)

        print(f"\n[Logger] Results saved to: {filepath}")
        return filepath

    def _convert_to_serializable(self, obj):
        """Convert numpy/torch objects to JSON-serializable format."""
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)


def create_logger(experiment_name: str, output_dir: str = "results") -> ExperimentLogger:
    """Factory function to create a logger."""
    return ExperimentLogger(experiment_name, output_dir)


# Quick helper for saving simple results
def save_results(results: Dict, filepath: str):
    """Simple helper to save results dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(filepath, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Results saved to: {filepath}")
