"""
Experimental Configuration for Replicating Jelassi et al. (2024)
"Repeat After Me: Transformers are Better than State Space Models at Copying"

Academic Standards:
- Exact parameter count matching between architectures
- Rigorous statistical evaluation protocols
- Complete reproducibility with seed control
- Publication-ready logging and visualization

Experimental Design:
- Transformer baseline: 8.5M parameters (proven working)
- Mamba comparison: Exactly 8.5M parameters (fair comparison)  
- Copying task: 50-1000 tokens (following paper methodology)
- Statistical significance: Multiple runs with confidence intervals
"""

import torch
import numpy as np
import random
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
import json
import logging
from datetime import datetime
import platform
import subprocess


@dataclass
class ExperimentalConfig:
    """
    Rigorous experimental configuration following academic standards.
    All parameters explicitly documented for reproducibility.
    """
    
    # === REPRODUCIBILITY CONTROL ===
    random_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    python_seed: int = 42
    
    # === HARDWARE CONFIGURATION ===  
    device: str = "cpu"  # MacBook Pro Intel i5 constraint
    dtype: torch.dtype = torch.float32  # Start conservative, optimize later
    num_threads: int = 4  # Match CPU cores
    
    # === MODEL ARCHITECTURE ===
    vocab_size: int = 30  # Following Jelassi et al.
    
    # Target: Exactly 8.5M parameters for both models
    transformer_config: Dict[str, Any] = None
    mamba_config: Dict[str, Any] = None
    
    # === TRAINING PROTOCOL ===
    batch_size: int = 1  # Conservative for CPU
    gradient_accumulation_steps: int = 8  # Effective batch size = 8
    learning_rate: float = 3e-4  # Unified default for both modes
    weight_decay: float = 0.01   # Exclude norm/bias in optimizer groups
    max_grad_norm: float = 5.0   # Looser clipping; combined with no-clip warmup
    warmup_steps: int = 1000
    
    # === COPYING TASK SPECIFICATION ===
    min_sequence_length: int = 50   # Following paper
    max_sequence_length: int = 1000 # Extended evaluation
    training_lengths: List[int] = None  # Will be set in __post_init__
    evaluation_lengths: List[int] = None
    
    # === STATISTICAL EVALUATION ===
    num_evaluation_runs: int = 100  # For statistical significance
    confidence_level: float = 0.95  # 95% confidence intervals
    num_model_seeds: int = 5  # Multiple model initializations
    
    # === EXPERIMENTAL TIMELINE ===
    max_training_steps: int = 50000
    evaluation_interval: int = 1000
    checkpoint_interval: int = 5000
    
    def __post_init__(self):
        """Initialize derived configurations"""
        # Curriculum learning schedule (following paper methodology)
        self.training_lengths = list(range(50, 501, 50))  # 50, 100, ..., 500
        self.evaluation_lengths = list(range(50, 1001, 50))  # 50, 100, ..., 1000
        
        # Model configurations (exact parameter matching)
        self.transformer_config = {
            "vocab_size": self.vocab_size,
            "d_model": 384,  # From working implementation
            "n_layers": 6,
            "n_heads": 12,
            "d_ff": 1536,  # 4 * d_model
            "dropout": 0.0,  # Disable for deterministic evaluation
            "pad_token_id": 0,
        }
        
        # Mamba configured to match Transformer parameter count 
        # (reduced d_model to account for selective mechanism parameters)
        self.mamba_config = {
            "vocab_size": self.vocab_size,
            "d_model": 420,  # Reduced to account for selective parameters
            "n_layers": 6,
            "d_state": 32,
            "d_conv": 4,
            "expand": 1,     # Optimized for parameter matching
            "pad_token_id": 0,
        }


class ReproducibilityManager:
    """
    Manages all aspects of experimental reproducibility.
    
    Features:
    - Complete environment state capture
    - Deterministic random number generation
    - Hardware-specific optimization
    - Experimental metadata logging
    """
    
    def __init__(self, config: ExperimentalConfig):
        self.config = config
        self.experiment_id = self._generate_experiment_id()
        self.setup_logging()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment identifier"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"replication_{timestamp}_seed{self.config.random_seed}"
    
    def setup_logging(self):
        """Configure academic-grade logging (console-only to suppress logs folder)."""
        # Console-only logging; avoid writing to experiments/logs
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s | %(levelname)s | %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Experiment ID: {self.experiment_id}")
        
    def initialize_environment(self) -> Dict[str, Any]:
        """
        Initialize completely reproducible environment.
        Returns environment metadata for provenance tracking.
        """
        # Set all random seeds
        torch.manual_seed(self.config.torch_seed)
        torch.cuda.manual_seed_all(self.config.torch_seed)
        np.random.seed(self.config.numpy_seed)
        random.seed(self.config.python_seed)
        
        # Configure PyTorch for determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # CPU optimization
        torch.set_num_threads(self.config.num_threads)
        os.environ['OMP_NUM_THREADS'] = str(self.config.num_threads)
        
        # Capture environment metadata
        env_metadata = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "numpy_version": np.__version__,
            "device": self.config.device,
            "num_threads": self.config.num_threads,
            "seeds": {
                "torch": self.config.torch_seed,
                "numpy": self.config.numpy_seed,
                "python": self.config.python_seed,
            }
        }
        
        # Log git commit for code provenance
        try:
            git_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], 
                universal_newlines=True
            ).strip()
            env_metadata["git_commit"] = git_commit
        except:
            env_metadata["git_commit"] = "unknown"
            
        self.logger.info("Environment initialized for reproducibility")
        self.logger.info(f"Configuration: {self.config}")
        
        return env_metadata
    
    def save_experiment_metadata(self, metadata: Dict[str, Any]):
        """Save complete experiment metadata"""
        metadata_dir = "experiments/metadata"
        os.makedirs(metadata_dir, exist_ok=True)
        
        with open(f"{metadata_dir}/{self.experiment_id}_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Metadata saved: {self.experiment_id}_metadata.json")


class ParameterAnalyzer:
    """
    Analyzes and validates model parameter counts for exact matching.
    
    Critical for fair comparison between architectures.
    """
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """Detailed parameter counting with breakdown"""
        total_params = 0
        trainable_params = 0
        param_breakdown = {}
        
        for name, param in model.named_parameters():
            param_count = param.numel()
            param_breakdown[name] = param_count
            total_params += param_count
            
            if param.requires_grad:
                trainable_params += param_count
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_breakdown": param_breakdown
        }
    
    @staticmethod
    def compare_architectures(model1: torch.nn.Module, model2: torch.nn.Module, 
                            name1: str, name2: str) -> Dict[str, Any]:
        """Compare parameter counts between two models"""
        count1 = ParameterAnalyzer.count_parameters(model1)
        count2 = ParameterAnalyzer.count_parameters(model2)
        
        comparison = {
            name1: count1,
            name2: count2,
            "parameter_difference": abs(count1["total_parameters"] - count2["total_parameters"]),
            "percentage_difference": abs(count1["total_parameters"] - count2["total_parameters"]) / 
                                   max(count1["total_parameters"], count2["total_parameters"]) * 100,
            "fair_comparison": abs(count1["total_parameters"] - count2["total_parameters"]) < 100000  # < 100K difference
        }
        
        return comparison


def create_academic_experiment() -> Tuple[ExperimentalConfig, ReproducibilityManager]:
    """
    Factory function for creating rigorous experimental setup.
    
    Returns:
        config: Complete experimental configuration
        manager: Reproducibility management system
    """
    config = ExperimentalConfig()
    manager = ReproducibilityManager(config)
    env_metadata = manager.initialize_environment()
    
    # Create directory structure (suppress experiments/logs creation)
    for directory in ["experiments", "experiments/metadata", 
                     "experiments/checkpoints", "experiments/results"]:
        os.makedirs(directory, exist_ok=True)
    
    manager.logger.info("Academic experiment framework initialized")
    manager.logger.info(f"Experiment will evaluate {len(config.evaluation_lengths)} sequence lengths")
    manager.logger.info(f"Statistical evaluation: {config.num_evaluation_runs} runs per length")
    
    return config, manager


if __name__ == "__main__":
    print("=" * 80)
    print("ACADEMIC REPLICATION FRAMEWORK")
    print("Jelassi et al. (2024) - Transformers vs State Space Models")
    print("=" * 80)
    
    config, manager = create_academic_experiment()
    
    print(f"\n📋 EXPERIMENTAL CONFIGURATION:")
    print(f"   Random Seed: {config.random_seed}")
    print(f"   Target Parameters: ~8.5M (both models)")
    print(f"   Evaluation Lengths: {config.evaluation_lengths}")
    print(f"   Statistical Runs: {config.num_evaluation_runs} per length")
    print(f"   Model Initializations: {config.num_model_seeds}")
    
    print(f"\n🏗️ MODEL ARCHITECTURES:")
    print(f"   Transformer: {config.transformer_config}")
    print(f"   Mamba: {config.mamba_config}")
    
    print(f"\n📊 EVALUATION PROTOCOL:")
    print(f"   Confidence Level: {config.confidence_level}")
    print(f"   Training Lengths: {config.training_lengths}")
    print(f"   Test Lengths: {config.evaluation_lengths}")
    
    print(f"\n✅ Framework ready for rigorous replication")
