"""
Training utilities for the academic replication.
Clean, focused training functions following ML research standards.
Enhanced with comprehensive and EXTREME debugging to identify training failures.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import math


# ============================================================================
# EXTREME debugging utilities (opt-in, very verbose)
# ============================================================================

# Global flag for extreme verbosity
EXTREME_VERBOSITY = False

# Optional global logger for EXTREME logs
EXTREME_LOGGER: Optional[logging.Logger] = None

def set_extreme_logger(logger: Optional[logging.Logger]):
    """Attach/detach a global logger used by extreme logging helpers."""
    global EXTREME_LOGGER
    EXTREME_LOGGER = logger

def set_extreme_verbosity(enabled: bool):
    """Enable/disable extreme verbosity globally (silent toggle)."""
    global EXTREME_VERBOSITY
    EXTREME_VERBOSITY = bool(enabled)


def extreme_log(message: str, data: Any = None, force: bool = False):
    """Log with extreme verbosity if enabled"""
    if EXTREME_VERBOSITY or force:
        _line = f"🔍 {message}"
        print(_line)
        if EXTREME_LOGGER is not None:
            EXTREME_LOGGER.info(_line)
        if data is not None:
            if isinstance(data, torch.Tensor):
                try:
                    _line = f"   Shape: {tuple(data.shape)}, Device: {data.device}, Dtype: {data.dtype}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
                    _line = f"   Range: [{data.min().item():.6f}, {data.max().item():.6f}]"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
                    _line = f"   Mean: {data.mean().item():.6f}, Std: {data.std().item():.6f}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
                except Exception:
                    pass
                if data.numel() <= 20:
                    _line = f"   Values: {data.tolist()}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
                else:
                    flat = data.detach().reshape(-1)
                    first = flat[:10].tolist()
                    last = flat[-10:].tolist()
                    _line = f"   First 10: {first}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
                    _line = f"   Last 10: {last}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
            elif isinstance(data, (list, tuple)) and len(data) <= 20:
                _line = f"   Data: {data}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
            elif isinstance(data, dict):
                for k, v in data.items():
                    _line = f"   {k}: {v}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)
            else:
                _line = f"   Data: {data}"; print(_line); EXTREME_LOGGER and EXTREME_LOGGER.info(_line)



class ComprehensiveTrainer:
    """
    Gold standard training class following ML research best practices.
    
    Features:
    - Real-time loss/metric tracking with validation curves
    - Gradient norm monitoring for training stability  
    - Parameter norm tracking for model health
    - Learning rate scheduling with warmup + cosine decay
    - Early stopping with patience
    - Comprehensive logging (console + file + plots)
    - Model checkpointing at regular intervals
    - Per-sequence-length validation during training
    - Memory and timing profiling
    - Statistical significance testing
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], model_name: str = "Model", 
                 log_dir: Optional[str] = None):
        self.model = model
        self.config = config
        self.model_name = model_name
        self.device = config.get("device", "cpu")
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup logging directory
        self.log_dir = Path(log_dir) if log_dir else Path(f"experiments/training_logs/{model_name}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup file logging
        self.setup_logging()
        
        # Optimizer with proper hyperparameters
        base_lr = config.get("learning_rate", 2e-4)
        # Use a conservative default; caller can override via config
        weight_decay = config.get("weight_decay", 0.01)
        
        # Exclude LayerNorm/Norm and bias parameters from weight decay
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            if lname.endswith('bias') or 'norm' in lname or 'layernorm' in lname or '.ln' in lname:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0},
        ]
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=base_lr,
            betas=(0.9, 0.95),  # Standard for language models
            eps=1e-8
        )
        
        # Learning rate scheduler
        # Warmup measured in optimizer steps; ensure at least 500 for stability
        try:
            _ws = int(config.get("warmup_steps", 500))
        except Exception:
            _ws = 500
        self.warmup_steps = max(500, _ws)
        self.max_steps = config.get("max_training_steps", 50000)
        # Gradient clipping controls
        self.max_grad_norm = float(config.get("max_grad_norm", 5.0))  # loosen to 5.0 by default
        # Default: no delay in clipping to prevent early instability
        self.no_clip_steps = int(config.get("no_clip_steps", 0))

        # Extreme logging setup (file logger initialized lazily when used)
        self.extreme_logger: Optional[logging.Logger] = None
        
        # Training metrics with comprehensive tracking
        self.training_history = {
            'steps': [],
            'train_losses': [],
            'val_losses': [],
            'val_accuracies': {},  # per sequence length
            'learning_rates': [],
            'gradient_norms': [],
            'parameter_norms': [],
            'training_time': [],
            'memory_usage': [],
            'tokens_per_second': []
        }
        
        # Intelligent early stopping based on training progress
        self.best_val_loss = float('inf')
        # Track token accuracy as primary signal for progress on copying
        self.best_val_token_accuracy = 0.0
        self.loss_plateau_patience = config.get("loss_plateau_patience", 1000)  # Steps without loss improvement
        self.accuracy_stuck_patience = config.get("accuracy_stuck_patience", 2000)  # Steps at low token accuracy
        self.loss_plateau_threshold = config.get("loss_plateau_threshold", 1e-4)  # Minimum loss improvement
        self.accuracy_improvement_threshold = config.get("accuracy_improvement_threshold", 0.01)  # 1% absolute token acc improvement
        self.token_acc_stuck_threshold = config.get("token_acc_stuck_threshold", 0.05)  # <=5% token acc considered stuck
        
        # Progress tracking
        # Maintain counters in units of validations; convert to steps when logging
        self.evals_without_loss_improvement = 0
        self.evals_at_low_accuracy = 0
        self.evals_without_accuracy_improvement = 0
        self.loss_history_window = []  # Track recent losses for plateau detection
        self.accuracy_history_window = []  # Track recent token accuracies
        
        # Validation setup
        self.val_interval = config.get("validation_interval", 200)
        self.val_lengths = config.get("validation_lengths", [50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
        
        # Debug sequence processing on first example
        self._debug_sequences_logged = False
        self.val_samples = config.get("validation_samples", 20)
        
        # Checkpointing
        self.checkpoint_interval = config.get("checkpoint_interval", 2000)
        self.checkpoint_dir = self.log_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.start_time = None
        self.tokens_processed = 0
        
        self.logger.info(f"ComprehensiveTrainer initialized for {model_name}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        self.logger.info(f"Log directory: {self.log_dir}")
        
    def setup_logging(self):
        """Setup comprehensive logging system"""
        # Create logger
        self.logger = logging.getLogger(f"trainer_{self.model_name}")
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # File handler
        file_handler = logging.FileHandler(self.log_dir / f"{self.model_name}_training.log")
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
    # Removed reverse-order logging method as requested

    # ===== EXTREME logging helpers (no computation changes) =====
    def _setup_extreme_logger(self):
        if self.extreme_logger is not None:
            return
        xlog_dir = self.log_dir
        xlog_dir.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger(f"extreme_{self.model_name}")
        logger.setLevel(logging.INFO)
        logger.handlers = []
        fh = logging.FileHandler(xlog_dir / "extreme_debug.log")
        fmt = logging.Formatter('%(asctime)s | %(message)s')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        self.extreme_logger = logger

    def _xlog(self, msg: str):
        # Console and extreme file
        print(msg)
        try:
            if self.extreme_logger is not None:
                self.extreme_logger.info(msg)
        except Exception:
            pass

    def compute_gradient_diagnostics_extreme(self) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {
            'total_norm': 0.0,
            'max_grad': 0.0,
            'min_grad': float('inf'),
            'num_zero_grads': 0,
            'num_large_grads': 0,
            'layer_norms': {}
        }
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                g = p.grad.detach()
                gnorm = g.data.norm(2).item()
                gmax = g.data.abs().max().item()
                try:
                    gmin = g.data.abs().min().item()
                except Exception:
                    gmin = 0.0
                diagnostics['total_norm'] += gnorm ** 2
                diagnostics['max_grad'] = max(diagnostics['max_grad'], gmax)
                diagnostics['min_grad'] = min(diagnostics['min_grad'], gmin)
                lname = name.split('.')[0]
                diagnostics['layer_norms'].setdefault(lname, 0.0)
                diagnostics['layer_norms'][lname] += gnorm ** 2
                extreme_log(f"Gradient for {name}", {
                    'norm': gnorm,
                    'max': gmax,
                    'min': gmin,
                    'shape': list(g.shape)
                })
        diagnostics['total_norm'] = diagnostics['total_norm'] ** 0.5
        for lname in list(diagnostics['layer_norms'].keys()):
            diagnostics['layer_norms'][lname] = diagnostics['layer_norms'][lname] ** 0.5
        if diagnostics['min_grad'] == float('inf'):
            diagnostics['min_grad'] = 0.0
        return diagnostics

    def train_step_with_extreme_logging(self, input_tensor: torch.Tensor, target_tensor: torch.Tensor,
                                        task, step: int) -> Dict[str, float]:
        """Training step with optional EXTREME logging.

        If EXTREME_VERBOSITY is enabled, delegates to ExtremeDebugTrainer for full diagnostics.
        Otherwise performs a regular teacher-forced training step with minimal logging.
        """

        if EXTREME_VERBOSITY:
            temp_trainer = ExtremeDebugTrainer(self.model, self.config, self.model_name)
            # Share optimizer and step counter for consistency
            temp_trainer.optimizer = self.optimizer
            temp_trainer.total_steps = step
            return temp_trainer.train_step_extreme_debug(input_tensor, target_tensor, task, step)
        else:
            # Regular training step
            self.model.train()

            input_list = input_tensor[0].tolist()
            target_list = target_tensor[0].tolist()

            # Create full sequence for teacher forcing
            full_sequence = input_list + target_list
            full_tensor = torch.tensor(full_sequence, device=self.device).unsqueeze(0)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(full_tensor)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs

            # Compute loss on target part
            target_start = len(input_list)
            if target_start + len(target_list) <= logits.size(1):
                pred_logits = logits[0, target_start-1:target_start-1+len(target_list)]
                target_tokens = torch.tensor(target_list, device=self.device)
                loss = nn.CrossEntropyLoss()(pred_logits, target_tokens)

                # Backward pass
                loss.backward()
                grad_norm = self.compute_gradient_norm()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get("max_grad_norm", 1.0))

                # Optimizer step
                current_lr = self.update_learning_rate(step)
                self.optimizer.step()

                return {
                    "loss": float(loss.item()),
                    "lr": float(current_lr),
                    "grad_norm": float(grad_norm),
                    "param_norm": float(self.compute_parameter_norm())
                }

            return {"loss": float('inf'), "lr": 0.0, "grad_norm": 0.0, "param_norm": 0.0}
        
    def get_learning_rate(self, step: int) -> float:
        """Learning rate with warmup + cosine decay"""
        base_lr = self.config.get("learning_rate", 2e-4)
        min_lr = self.config.get("min_learning_rate", base_lr * 0.1)
        
        if step < self.warmup_steps:
            # Linear warmup
            return base_lr * (step + 1) / self.warmup_steps
        else:
            # Cosine annealing
            progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return min_lr + (base_lr - min_lr) * cosine_decay
    
    def update_learning_rate(self, step: int):
        """Update optimizer learning rate"""
        lr = self.get_learning_rate(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def compute_gradient_norm(self) -> float:
        """Compute gradient norm for monitoring training stability"""
        total_norm = 0.0
        param_count = 0
        
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        return (total_norm ** 0.5) if param_count > 0 else 0.0
    
    def compute_parameter_norm(self) -> float:
        """Compute parameter norm for model health monitoring"""
        total_norm = 0.0
        for p in self.model.parameters():
            param_norm = p.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
        return total_norm ** 0.5
    
    def validate_model(self, task, step: int) -> Dict[str, float]:
        """Comprehensive validation across multiple sequence lengths"""
        self.model.eval()
        val_results = {}
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for length in self.val_lengths:
                correct = 0
                length_loss = 0.0
                
                for _ in range(self.val_samples):
                    try:
                        input_seq, target_seq = task.create_copy_example(length, "uniform")
                        
                        # Convert to tensors
                        if isinstance(input_seq, torch.Tensor):
                            input_tensor = input_seq.unsqueeze(0)
                            target_tensor = target_seq.unsqueeze(0)
                        else:
                            input_tensor = torch.tensor(input_seq).unsqueeze(0)
                            target_tensor = torch.tensor(target_seq).unsqueeze(0)
                        
                        # Create full sequence for evaluation
                        input_list = input_tensor[0].tolist()
                        target_list = target_tensor[0].tolist()
                        full_sequence = input_list + target_list
                        full_tensor = torch.tensor(full_sequence, device=self.device).unsqueeze(0)
                        
                        outputs = self.model(full_tensor)
                        logits = outputs["logits"]
                        
                        # Compute loss on target part
                        target_start = len(input_list)
                        if target_start + len(target_list) <= logits.size(1):
                            pred_logits = logits[0, target_start-1:target_start-1+len(target_list)]
                            target_tokens = torch.tensor(target_list, device=self.device)
                            loss = nn.CrossEntropyLoss()(pred_logits, target_tokens)
                            length_loss += loss.item()
                            
                            # Check sequence-level accuracy (exact match)
                            predictions = pred_logits.argmax(dim=-1).tolist()
                            if predictions == target_list:
                                correct += 1
                            
                            # Also track token-level accuracy for progress monitoring
                            if not hasattr(self, 'token_correct'):
                                self.token_correct = {}
                                self.token_total = {}
                            if length not in self.token_correct:
                                self.token_correct[length] = 0
                                self.token_total[length] = 0
                            
                            token_matches = sum(1 for p, t in zip(predictions, target_list) if p == t)
                            self.token_correct[length] += token_matches
                            self.token_total[length] += len(target_list)
                    
                    except Exception as e:
                        self.logger.warning(f"Validation error at length {length}: {e}")
                        continue
                
                accuracy = correct / self.val_samples
                avg_loss = length_loss / self.val_samples
                val_results[f"val_acc_{length}"] = accuracy
                val_results[f"val_loss_{length}"] = avg_loss
                
                # Calculate token-level accuracy for this validation round
                if hasattr(self, 'token_correct') and length in self.token_correct:
                    token_acc = self.token_correct[length] / max(self.token_total[length], 1)
                    val_results[f"val_token_acc_{length}"] = token_acc
                    # Reset counters for next validation
                    self.token_correct[length] = 0
                    self.token_total[length] = 0
                
                total_loss += avg_loss
                total_samples += 1
        
        # Overall validation loss
        val_results["val_loss_overall"] = total_loss / max(total_samples, 1)
        
        # Log validation results
        self.logger.info(f"Step {step} Validation:")
        for length in self.val_lengths:
            acc = val_results.get(f"val_acc_{length}", 0.0)
            token_acc = val_results.get(f"val_token_acc_{length}", 0.0)
            loss = val_results.get(f"val_loss_{length}", float('inf'))
            self.logger.info(f"  Length {length}: {acc:.1%} seq_acc, {token_acc:.1%} token_acc, {loss:.4f} loss")
        
        self.model.train()
        return val_results
    
    def check_early_stopping(self, val_results: Dict[str, float], step: int) -> Dict[str, bool]:
        """
        Intelligent early stopping based on actual training progress.
        
        Returns dictionary with stopping decisions and reasons.
        """
        val_loss = val_results.get("val_loss_overall", float('inf'))
        
        # Primary progress signal: token-level accuracy at length 50
        val_token_accuracy = val_results.get("val_token_acc_50", 0.0)
        
        stopping_info = {
            'should_stop': False,
            'reason': None,
            'loss_plateau': False,
            'accuracy_stuck': False,
            'accuracy_plateau': False
        }
        
        # Update loss tracking
        self.loss_history_window.append(val_loss)
        if len(self.loss_history_window) > 20:  # Keep last 20 validations
            self.loss_history_window.pop(0)
        
        # Update accuracy tracking (token accuracy)
        self.accuracy_history_window.append(val_token_accuracy)
        if len(self.accuracy_history_window) > 20:
            self.accuracy_history_window.pop(0)
        
        # Convert patience from steps to number of validations
        patience_evals_loss = max(1, math.ceil(self.loss_plateau_patience / self.val_interval))
        patience_evals_acc = max(1, math.ceil(self.accuracy_stuck_patience / self.val_interval))
        
        # Check loss improvement
        if val_loss < self.best_val_loss - self.loss_plateau_threshold:
            self.best_val_loss = val_loss
            self.evals_without_loss_improvement = 0
            self.logger.info(f"New best validation loss: {val_loss:.6f}")
        else:
            self.evals_without_loss_improvement += 1
        
        # Check token accuracy improvement
        if val_token_accuracy > self.best_val_token_accuracy + self.accuracy_improvement_threshold:
            self.best_val_token_accuracy = val_token_accuracy
            self.evals_without_accuracy_improvement = 0
            self.evals_at_low_accuracy = 0
            self.logger.info(f"New best token accuracy: {val_token_accuracy:.1%}")
        else:
            self.evals_without_accuracy_improvement += 1
            
            # Track low token accuracy specifically
            if val_token_accuracy <= self.token_acc_stuck_threshold:
                self.evals_at_low_accuracy += 1
            else:
                self.evals_at_low_accuracy = 0
        
        # Early stopping conditions
        # 1) Loss plateau: loss hasn't improved for patience_evals_loss validations
        if self.evals_without_loss_improvement >= patience_evals_loss:
            stopping_info['loss_plateau'] = True
        
        # 2) Token accuracy stuck low AND loss plateau
        if (self.evals_at_low_accuracy >= patience_evals_acc and
            self.evals_without_loss_improvement >= patience_evals_loss):
            stopping_info['accuracy_stuck'] = True
        
        # 3) Accuracy plateau: token accuracy flat for a while (but not low)
        if (self.evals_without_accuracy_improvement >= patience_evals_loss and 
            val_token_accuracy > self.token_acc_stuck_threshold and len(self.accuracy_history_window) >= 10):
            recent_accuracies = self.accuracy_history_window[-10:]
            if max(recent_accuracies) - min(recent_accuracies) < 0.02:  # <2% variation
                stopping_info['accuracy_plateau'] = True
        
        # Decide whether to stop based on multiple criteria
        if stopping_info['accuracy_stuck']:
            stopping_info['should_stop'] = True
            steps_low_acc = self.evals_at_low_accuracy * self.val_interval
            steps_loss_plateau = self.evals_without_loss_improvement * self.val_interval
            stopping_info['reason'] = (
                f"Token accuracy â‰¤ {self.token_acc_stuck_threshold:.0%} and loss plateaued "
                f"(low-acc steps: {steps_low_acc}, loss plateau steps: {steps_loss_plateau})"
            )
        elif stopping_info['loss_plateau'] and stopping_info['accuracy_plateau']:
            stopping_info['should_stop'] = True
            steps_loss_plateau = self.evals_without_loss_improvement * self.val_interval
            steps_acc_plateau = self.evals_without_accuracy_improvement * self.val_interval
            stopping_info['reason'] = (
                f"Both loss and token accuracy plateaued (loss: {steps_loss_plateau} steps, "
                f"acc: {steps_acc_plateau} steps)"
            )
        elif stopping_info['loss_plateau'] and val_token_accuracy > 0.5:  # Good accuracy, loss plateaued
            stopping_info['should_stop'] = True
            steps_loss_plateau = self.evals_without_loss_improvement * self.val_interval
            stopping_info['reason'] = f"Loss plateaued at good token accuracy ({val_token_accuracy:.1%}) for {steps_loss_plateau} steps"
        
        # Log progress information periodically
        if step % (self.val_interval * 2) == 0:
            self.logger.info(f"Training Progress Monitor:")
            self.logger.info(f"  Steps without loss improvement: {self.evals_without_loss_improvement * self.val_interval}")
            self.logger.info(f"  Steps at low token accuracy (â‰¤{self.token_acc_stuck_threshold:.0%}): {self.evals_at_low_accuracy * self.val_interval}")
            self.logger.info(f"  Steps without token accuracy improvement: {self.evals_without_accuracy_improvement * self.val_interval}")
            self.logger.info(f"  Best val loss: {self.best_val_loss:.6f}")
            self.logger.info(f"  Best token accuracy: {self.best_val_token_accuracy:.1%}")
        
        return stopping_info
    
    def debug_sequence_processing(self, task, input_seq, target_seq, step):
        """Debug and log sequence processing details"""
        if self._debug_sequences_logged:
            return  # Only log once
        
        self.logger.info(f"=== SEQUENCE PROCESSING DEBUG (Step {step}) ===")
        self.logger.info(f"BOS_TOKEN = {task.BOS_TOKEN}")
        self.logger.info(f"COPY_TOKEN = {task.COPY_TOKEN}")
        self.logger.info(f"EOS_TOKEN = {task.EOS_TOKEN}")
        
        # Convert tensors if needed
        if isinstance(input_seq, torch.Tensor):
            input_list = input_seq.tolist()
            target_list = target_seq.tolist()
        else:
            input_list = input_seq
            target_list = target_seq
        
        self.logger.info(f"Example sequence (length={len(target_list)-1}):")  # -1 for EOS
        self.logger.info(f"  input_seq  = {input_list}")
        self.logger.info(f"  target_seq = {target_list}")
        
        # Full sequence construction
        full_sequence = input_list + target_list
        self.logger.info(f"  full_sequence = {full_sequence}")
        
        # Position analysis
        self.logger.info("Position analysis:")
        self.logger.info(f"  Positions: {list(range(len(full_sequence)))}")
        self.logger.info(f"  Tokens:    {full_sequence}")
        
        # Training computation analysis
        target_start = len(input_list)
        pred_positions = list(range(target_start-1, target_start-1+len(target_list)))
        
        self.logger.info(f"Training loss computation:")
        self.logger.info(f"  len(input_list) = {len(input_list)}")
        self.logger.info(f"  len(target_list) = {len(target_list)}")
        self.logger.info(f"  target_start = {target_start}")
        self.logger.info(f"  Prediction positions: {pred_positions}")
        
        # Detailed position analysis
        self.logger.info("Position-by-position prediction analysis:")
        for i, pos in enumerate(pred_positions):
            if pos+1 < len(full_sequence):
                expected_next = full_sequence[pos+1]
                target_token = target_list[i]
                match = "âœ“" if expected_next == target_token else "âœ—"
                self.logger.info(f"    Pos {pos}: predict next={expected_next}, target={target_token} {match}")
        
        # Sequence interpretation
        self.logger.info("Sequence token breakdown:")
        for i, token in enumerate(full_sequence):
            if token == task.BOS_TOKEN:
                desc = "BOS (start of input)"
            elif token == task.COPY_TOKEN:
                desc = "COPY (copy signal)"
            elif token == task.EOS_TOKEN:
                desc = "EOS (end of target)"
            else:
                desc = f"regular token"
            self.logger.info(f"    Position {i}: token {token} ({desc})")
        
        self._debug_sequences_logged = True
    
    def save_checkpoint(self, step: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'training_history': self.training_history,
            'config': self.config
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if val_loss < self.best_val_loss:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best checkpoint saved: {val_loss:.6f}")
    
    def plot_training_curves(self, save_path: Optional[str] = None):
        """Generate comprehensive training plots"""
        if not self.training_history['steps']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_name} Training Progress', fontsize=16)
        
        steps = self.training_history['steps']
        
        # Loss curves
        axes[0, 0].plot(steps, self.training_history['train_losses'], label='Train Loss', alpha=0.7)
        if self.training_history['val_losses']:
            # Create validation steps based on actual validation intervals
            num_vals = len(self.training_history['val_losses'])
            val_steps = [self.val_interval * (i + 1) for i in range(num_vals)]
            axes[0, 0].plot(val_steps, self.training_history['val_losses'], label='Val Loss', alpha=0.9)
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training/Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Learning rate
        axes[0, 1].plot(steps, self.training_history['learning_rates'], color='green')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        
        # Gradient norms
        axes[0, 2].plot(steps, self.training_history['gradient_norms'], color='orange', alpha=0.7)
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].set_ylabel('Gradient Norm')
        axes[0, 2].set_title('Gradient Norm (Training Stability)')
        axes[0, 2].set_yscale('log')
        
        # Parameter norms
        axes[1, 0].plot(steps, self.training_history['parameter_norms'], color='purple')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Parameter Norm')
        axes[1, 0].set_title('Model Parameter Norm')
        
        # Validation accuracies by length
        for length in self.val_lengths:
            key = f"val_acc_{length}"
            if key in self.training_history['val_accuracies']:
                num_vals = len(self.training_history['val_accuracies'][key])
                val_steps = [self.val_interval * (i + 1) for i in range(num_vals)]
                axes[1, 1].plot(val_steps, self.training_history['val_accuracies'][key], 
                               label=f'Length {length}', alpha=0.8)
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Validation Accuracy by Sequence Length')
        axes[1, 1].legend()
        
        # Training speed
        if self.training_history['tokens_per_second']:
            axes[1, 2].plot(steps, self.training_history['tokens_per_second'], color='red')
            axes[1, 2].set_xlabel('Steps')
            axes[1, 2].set_ylabel('Tokens/Second')
            axes[1, 2].set_title('Training Speed')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.log_dir / f"{self.model_name}_training_curves.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Training curves saved: {save_path}")
    
    def save_training_logs(self):
        """Save training history to JSON"""
        log_path = self.log_dir / f"{self.model_name}_training_history.json"
        
        # Convert numpy types to native Python types for JSON serialization
        history_json = {}
        for key, values in self.training_history.items():
            if isinstance(values, dict):
                history_json[key] = {k: [float(v) for v in vals] for k, vals in values.items()}
            else:
                history_json[key] = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in values]
        
        with open(log_path, 'w') as f:
            json.dump(history_json, f, indent=2)
        
        self.logger.info(f"Training history saved: {log_path}")


def train_model_comprehensive(model: nn.Module, model_name: str, task, config: Dict[str, Any], 
                             max_steps: int = None) -> Tuple[float, ComprehensiveTrainer]:
    """
    Gold standard training function following ML research best practices.
    
    This function implements comprehensive monitoring and logging recommended
    for academic research and production ML systems.
    
    Features:
    - Real-time loss curves (train/validation)
    - Gradient and parameter norm monitoring
    - Learning rate scheduling with warmup
    - Early stopping with validation tracking
    - Model checkpointing
    - Per-sequence-length performance tracking
    - Comprehensive logging and visualization
    
    Args:
        model: PyTorch model to train
        model_name: Model name for logging
        task: Training task (CopyingTask)
        config: Training configuration
        max_steps: Maximum training steps
        
    Returns:
        final_loss: Final validation loss
        trainer: ComprehensiveTrainer with full history
    """
    
    print(f"\n{'='*80}")
    print(f"🚀 COMPREHENSIVE TRAINING: {model_name}")
    print(f"{'='*80}")
    
    # Initialize comprehensive trainer
    trainer = ComprehensiveTrainer(model, config, model_name)
    
    # Training configuration
    batch_size = config.get("batch_size", 1)
    grad_accumulation = config.get("gradient_accumulation_steps", 8)
    max_grad_norm = trainer.max_grad_norm  # use trainer setting (default 5.0)
    
    # Curriculum learning setup
    min_length = config.get("min_sequence_length", 50)
    max_length = config.get("max_sequence_length", 300)
    
    # Set intelligent stopping parameters
    if max_steps is None:
        max_steps = 100000  # Safety fallback, but should stop via intelligent criteria
    
    trainer.logger.info(f"Intelligent Training Configuration:")
    trainer.logger.info(f"  Max steps (fallback): {max_steps}")
    trainer.logger.info(f"  Batch size: {batch_size}")
    trainer.logger.info(f"  Gradient accumulation: {grad_accumulation}")
    trainer.logger.info(f"  Max grad norm: {max_grad_norm}")
    trainer.logger.info(f"  No-clip warmup (opt steps): {trainer.no_clip_steps}")
    # Log sequence lengths (explicit list if provided)
    if "training_lengths" in config:
        trainer.logger.info(f"  Sequence lengths: {config['training_lengths']}")
    else:
        trainer.logger.info(f"  Sequence lengths: {min_length}-{max_length}")
    trainer.logger.info(f"  Loss plateau patience: {trainer.loss_plateau_patience} steps")
    trainer.logger.info(f"  Accuracy stuck patience: {trainer.accuracy_stuck_patience} steps")
    trainer.logger.info(f"  Stopping when loss plateaus AND token accuracy stays low")
    
    # Training loop with intelligent stopping
    trainer.start_time = time.time()
    step = 0  # micro-steps
    opt_step = 0  # optimizer steps (after gradient accumulation)
    accumulated_loss = 0.0
    last_logged_loss: Optional[float] = None
    
    # Use indefinite progress bar since we don't know when it will stop
    progress = tqdm(desc=f"Training {model_name} ")
    
    try:
        while step < max_steps:
            # Prepare extreme logger if needed and update LR
            if EXTREME_VERBOSITY and trainer.extreme_logger is None:
                trainer._setup_extreme_logger()
                set_extreme_logger(trainer.extreme_logger)
            # Update learning rate based on optimizer steps (not micro-steps)
            lr_current = trainer.update_learning_rate(opt_step)
            
            # Discrete curriculum learning: train extensively on each length before advancing
            # Allow explicit override via config['training_lengths']
            training_lengths = config.get("training_lengths", list(range(min_length, max_length + 1, 50)))
            # Ensure there is some overlap with validation lengths to surface early progress
            try:
                val_lens = list(trainer.val_lengths)
            except Exception:
                val_lens = []
            if isinstance(training_lengths, list):
                overlap = set(training_lengths).intersection(val_lens)
                if not overlap:
                    # Add a few short lengths commonly used in copying tasks
                    augment = [l for l in [10, 20, 40, 50, 80, 100, 150, 160] if l not in training_lengths]
                    if augment:
                        training_lengths = sorted(training_lengths + augment)
                        trainer.logger.info(f"Augmented curriculum to ensure overlap with validation: added {augment}")
            steps_per_length = max_steps // len(training_lengths)  # Equal time per length
            current_length_idx = min(step // steps_per_length, len(training_lengths) - 1)
            seq_length = training_lengths[current_length_idx]
            
            # Training step
            try:
                # Create training example
                input_seq, target_seq = task.create_copy_example(seq_length, "uniform")
                
                # Debug sequence processing on first example
                if step == 0:
                    trainer.debug_sequence_processing(task, input_seq, target_seq, step)
                
                # Convert to tensors
                if isinstance(input_seq, torch.Tensor):
                    input_tensor = input_seq.unsqueeze(0)
                    target_tensor = target_seq.unsqueeze(0)
                else:
                    input_tensor = torch.tensor(input_seq).unsqueeze(0)
                    target_tensor = torch.tensor(target_seq).unsqueeze(0)
                
                # EXTREME logging: inputs
                do_extreme = EXTREME_VERBOSITY and (step < 5 or step % 100 == 0)
                if do_extreme:
                    trainer._xlog(f"\n{'='*80}")
                    trainer._xlog(f"🔬 EXTREME DEBUG: TRAINING STEP {step}")
                    trainer._xlog(f"{'='*80}")
                    trainer._xlog("\n📊 STEP 1: INPUT ANALYSIS")
                    trainer._xlog("─" * 40)
                    extreme_log("Input tensor", input_tensor[0])
                    extreme_log("Target tensor", target_tensor[0])
                
                # Forward pass with teacher forcing
                input_list = input_tensor[0].tolist()
                target_list = target_tensor[0].tolist()
                full_sequence = input_list + target_list
                # Ensure tensor is on the correct device (important off-CPU)
                full_tensor = torch.tensor(full_sequence, device=trainer.device).unsqueeze(0)
                if do_extreme:
                    trainer._xlog("\n📊 STEP 2: TEACHER FORCING SEQUENCE CONSTRUCTION")
                    trainer._xlog("─" * 40)
                    trainer._xlog(f"   Full sequence length: {len(full_sequence)}")
                    trainer._xlog(f"   Full sequence: {full_sequence[:10]}...{full_sequence[-5:]}")
                    extreme_log("Full tensor for forward pass", full_tensor[0])
                
                model.train()
                outputs = model(full_tensor)
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs
                if do_extreme:
                    trainer._xlog("\n📊 STEP 3: FORWARD PASS")
                    trainer._xlog("─" * 40)
                    extreme_log("Output logits", logits)
                
                # Compute loss on target part only
                target_start = len(input_list)
                if target_start + len(target_list) <= logits.size(1):
                    pred_logits = logits[0, target_start-1:target_start-1+len(target_list)]
                    target_tokens = torch.tensor(target_list, device=trainer.device)
                    ce_loss = nn.CrossEntropyLoss()(pred_logits, target_tokens)
                    if do_extreme:
                        trainer._xlog("\n📊 STEP 4: LOSS COMPUTATION")
                        trainer._xlog("─" * 40)
                        trainer._xlog(f"   Target starts at position: {target_start}")
                        trainer._xlog(f"   We need to predict positions {target_start} to {target_start + len(target_list) - 1}")
                        trainer._xlog(f"   Using logits from positions {target_start-1} to {target_start-1+len(target_list)-1}")
                        extreme_log("Prediction logits", pred_logits)
                        extreme_log("Target tokens", target_tokens)
                        trainer._xlog(f"   Loss value: {ce_loss.item():.8f}")
                        trainer._xlog("\n📊 STEP 5: PREDICTION ANALYSIS")
                        trainer._xlog("─" * 40)
                        predictions = pred_logits.argmax(dim=-1)
                        extreme_log("Predicted tokens", predictions)
                        token_acc = float((predictions == target_tokens).float().mean().item())
                        unique_preds = len(set(predictions.tolist()))
                        probs = torch.softmax(pred_logits, dim=-1)
                        max_probs = probs.max(dim=-1)[0]
                        trainer._xlog(f"   Unique predicted tokens: {unique_preds}/{len(predictions)}")
                        trainer._xlog(f"   Average confidence: {max_probs.mean().item():.4f}")
                        trainer._xlog(f"   Min confidence: {max_probs.min().item():.4f}")
                        trainer._xlog(f"   Max confidence: {max_probs.max().item():.4f}")
                    # Backward pass (accumulated)
                    loss = ce_loss / grad_accumulation
                    if do_extreme:
                        trainer._xlog("\n📊 STEP 6: BACKWARD PASS")
                        trainer._xlog("─" * 40)
                        trainer._xlog(f"   Loss (before backward): {ce_loss.item():.8f}")
                    loss.backward()
                    
                    accumulated_loss += loss.item()
                    trainer.tokens_processed += len(target_list)
                    
                    # Gradient accumulation
                    if (step + 1) % grad_accumulation == 0:
                        # Gradient diagnostics and optional clipping
                        if do_extreme:
                            trainer._xlog("\n📊 STEP 7: GRADIENT ANALYSIS")
                            trainer._xlog("─" * 40)
                            grad_diag = trainer.compute_gradient_diagnostics_extreme()
                            trainer._xlog(f"   Total gradient norm: {grad_diag['total_norm']:.8f}")
                            trainer._xlog(f"   Max gradient: {grad_diag['max_grad']:.8f}")
                            trainer._xlog(f"   Min gradient: {grad_diag['min_grad']:.8f}")
                            trainer._xlog(f"   Zero gradients: {grad_diag['num_zero_grads']}")
                            trainer._xlog(f"   Large gradients (>10): {grad_diag['num_large_grads']}")
                            trainer._xlog("\n   Layer-wise gradient norms:")
                            for layer, norm in grad_diag['layer_norms'].items():
                                trainer._xlog(f"      {layer}: {norm:.8f}")
                        grad_norm = trainer.compute_gradient_norm()
                        if opt_step >= trainer.no_clip_steps:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            if do_extreme:
                                trainer._xlog("\n📊 STEP 8: GRADIENT CLIPPING")
                                trainer._xlog("─" * 40)
                                trainer._xlog(f"   Clipping to max norm: {max_grad_norm}")
                                gn_after = trainer.compute_gradient_norm()
                                trainer._xlog(f"   Gradient norm after clipping: {gn_after:.8f}")
                        else:
                            if do_extreme:
                                trainer._xlog("\n📊 STEP 8: GRADIENT CLIPPING")
                                trainer._xlog("─" * 40)
                                trainer._xlog(f"   Skipping clipping (opt_step < {trainer.no_clip_steps})")
                        
                        # Optimizer step
                        if do_extreme:
                            trainer._xlog("\n📊 STEP 9: OPTIMIZER STEP")
                            trainer._xlog("─" * 40)
                            try:
                                pnorm_before = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
                            except Exception:
                                pnorm_before = 0.0
                            trainer._xlog(f"   Learning rate: {lr_current:.8e}")
                            trainer._xlog(f"   Parameter norm before update: {pnorm_before:.8f}")
                        trainer.optimizer.step()
                        trainer.optimizer.zero_grad()
                        
                        # Capture and reset accumulated loss for logging on optimizer steps
                        last_logged_loss = accumulated_loss
                        
                        # Record metrics
                        trainer.training_history['steps'].append(step)
                        trainer.training_history['train_losses'].append(accumulated_loss)
                        trainer.training_history['learning_rates'].append(lr_current)
                        trainer.training_history['gradient_norms'].append(grad_norm)
                        trainer.training_history['parameter_norms'].append(trainer.compute_parameter_norm())
                        
                        # Training speed
                        elapsed_time = time.time() - trainer.start_time
                        tokens_per_sec = trainer.tokens_processed / max(elapsed_time, 1e-6)
                        trainer.training_history['tokens_per_second'].append(tokens_per_sec)
                        
                        # Reset accumulation
                        accumulated_loss = 0.0
                        
                        # Advance optimizer step and update LR for next cycle
                        opt_step += 1
                        lr_current = trainer.update_learning_rate(opt_step)
                        if do_extreme:
                            try:
                                pnorm_after = sum(p.data.norm(2).item()**2 for p in model.parameters())**0.5
                            except Exception:
                                pnorm_after = 0.0
                            trainer._xlog(f"   Parameter norm after update: {pnorm_after:.8f}")
                            trainer._xlog(f"   Parameter change: {abs(pnorm_after - pnorm_before):.8f}")
                            trainer._xlog("\n📊 STEP 10: STEP SUMMARY")
                            trainer._xlog("─" * 40)
                            display_loss = last_logged_loss * grad_accumulation
                            trainer._xlog(f"   Loss: {display_loss:.8f}")
                            if 'predictions' in locals():
                                trainer._xlog(f"   Token Accuracy: {token_acc:.2%}")
                                trainer._xlog(f"   Exact Match: {bool((predictions.tolist()==target_list))}")
                            trainer._xlog(f"   Learning Rate: {lr_current:.8e}")
                            trainer._xlog(f"   Gradient Norm: {grad_norm:.8f}")
                        
                        # Update progress bar
                        progress.set_postfix({
                            'loss': f"{trainer.training_history['train_losses'][-1]:.4f}",
                            'lr': f"{lr_current:.2e}",
                            'grad_norm': f"{grad_norm:.3f}",
                            'seq_len': seq_length,
                            'tok/s': f"{tokens_per_sec:.0f}"
                        })
            
            except Exception as e:
                trainer.logger.error(f"Training error at step {step}: {e}")
                continue
            
            # Quick loss logging every 5 steps (for all training lengths)
            loss_log_interval = config.get("loss_log_interval", 5)
            if step > 0 and step % loss_log_interval == 0:
                # Prefer the last optimizer-step loss snapshot to avoid logging zeros
                snapshot = last_logged_loss if last_logged_loss is not None else accumulated_loss
                current_loss = snapshot * grad_accumulation  # Unscale for logging
                trainer.logger.info(f"Step {step} Training Loss: {current_loss:.4f} (length {seq_length})")
            
            # Full validation and intelligent early stopping every 200 steps
            if step > 0 and step % trainer.val_interval == 0:
                val_results = trainer.validate_model(task, step)
                val_loss = val_results.get("val_loss_overall", float('inf'))
                
                # Track validation history
                trainer.training_history['val_losses'].append(val_loss)
                
                # Track per-length accuracies
                for length in trainer.val_lengths:
                    key = f"val_acc_{length}"
                    if key not in trainer.training_history['val_accuracies']:
                        trainer.training_history['val_accuracies'][key] = []
                    trainer.training_history['val_accuracies'][key].append(val_results.get(key, 0.0))
                
                # Intelligent early stopping check
                stopping_info = trainer.check_early_stopping(val_results, step)
                
                if stopping_info['should_stop']:
                    trainer.logger.info(f"ðŸ›‘ Intelligent early stopping triggered at step {step}")
                    trainer.logger.info(f"   Reason: {stopping_info['reason']}")
                    break
            
            # Checkpointing
            if step > 0 and step % trainer.checkpoint_interval == 0:
                val_loss = trainer.training_history['val_losses'][-1] if trainer.training_history['val_losses'] else float('inf')
                trainer.save_checkpoint(step, val_loss)
                
                # Save intermediate plots
                trainer.plot_training_curves()
            
            # CRITICAL: Increment step counter and update progress bar
            step += 1
            progress.update(1)
    
    except KeyboardInterrupt:
        trainer.logger.info(f"Training interrupted at step {step}")
    
    # Final validation
    trainer.logger.info("Performing final validation...")
    final_val_results = trainer.validate_model(task, step)
    final_val_loss = final_val_results.get("val_loss_overall", float('inf'))
    
    # Save final checkpoint
    trainer.save_checkpoint(step, final_val_loss)
    
    # Generate final plots and save logs
    trainer.plot_training_curves()
    trainer.save_training_logs()
    
    # Training summary
    trainer.logger.info(f"\n{'='*60}")
    trainer.logger.info(f"TRAINING COMPLETE: {model_name}")
    trainer.logger.info(f"{'='*60}")
    trainer.logger.info(f"Steps completed: {step + 1}")
    trainer.logger.info(f"Final train loss: {trainer.training_history['train_losses'][-1]:.6f}")
    trainer.logger.info(f"Final val loss: {final_val_loss:.6f}")
    trainer.logger.info(f"Best val loss: {trainer.best_val_loss:.6f}")
    trainer.logger.info(f"Training time: {time.time() - trainer.start_time:.1f}s")
    
    for length in trainer.val_lengths:
        final_acc = final_val_results.get(f"val_acc_{length}", 0.0)
        trainer.logger.info(f"Final accuracy (length {length}): {final_acc:.1%}")
    # Detach extreme logger if attached
    try:
        set_extreme_logger(None)
    except Exception:
        pass
    
    return final_val_loss, trainer
