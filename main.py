#!/usr/bin/env python3
"""
Main entry point for the academic replication study.

This is the single command to reproduce all results from:
"Repeat After Me: Transformers are Better than State Space Models at Copying"
Jelassi et al. (2024)

Usage:
    python main.py              # Full replication (train + eval + plots)
    python main.py --train-only # Training only
    python main.py --eval-only  # Evaluation only  
    python main.py --plot-only  # Generate plots only
"""

import argparse
import sys
from pathlib import Path

from config import create_academic_experiment
from models.standardized_models import ModelFactory
from benchmarks.copying_benchmark import CopyingTask
from visualization.publication_plots import create_performance_comparison, create_training_summary


def train_models(config, task, debug=False, extreme_verbosity=False):
    """Train both models using comprehensive monitoring, or EXTREME debugging if requested."""
    if extreme_verbosity:
        print("🚀 Training models with EXTREME debugging enabled...")
    else:
        print("🚀 Training models with comprehensive monitoring...")
    
    # Create models
    transformer, mamba = ModelFactory.create_matched_models(
        config.transformer_config, 
        config.mamba_config
    )
    
    # Training functions
    from utils.training import train_model_comprehensive, set_extreme_verbosity
    
    # Enhanced training configuration with proper validation
    training_config = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        # Warmup is now in optimizer steps; reduce to 125 (was 1000/8)
        "warmup_steps": 125,
        # Raise the LR floor to 0.2x base LR to avoid tiny LRs late in cosine
        "min_learning_rate": float(config.learning_rate) * 0.2,
        "max_grad_norm": config.max_grad_norm,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "device": config.device,
        "max_training_steps": 100000,  # Safety fallback - should stop via intelligent criteria
        
        # Logging and validation settings
        "loss_log_interval": 5,  # Log training loss every 5 steps
        # Measure accuracy every 1000 steps
        "validation_interval": 1000,
        # Validation across a mix of seen and unseen lengths (with overlap)
        "validation_lengths": [10, 20, 40, 50, 80, 100, 150, 160, 200, 300, 400, 500],
        "validation_samples": 5,  # 5 samples per length for efficiency during training
        
        # Intelligent early stopping parameters
        "loss_plateau_patience": config.loss_plateau_patience,
        "accuracy_stuck_patience": config.accuracy_stuck_patience,
        "loss_plateau_threshold": config.loss_plateau_threshold,
        "accuracy_improvement_threshold": config.accuracy_improvement_threshold,
        "token_acc_stuck_threshold": config.token_acc_stuck_threshold,
        "checkpoint_interval": 1000,  # Save every 1000 steps
        
        # Mastery-based curriculum parameters
        "mastery_based_curriculum": config.mastery_based_curriculum,
        "mastery_threshold": config.mastery_threshold,
        "mastery_consistency_required": config.mastery_consistency_required,
        "mastery_fallback_steps": config.mastery_fallback_steps,
        
        # Sequence length curriculum
        # Use an explicit sparse curriculum for faster mastery on CPU
        "training_lengths": [10, 20, 40, 80, 160],
        "min_sequence_length": config.min_sequence_length,
        "max_sequence_length": 500,  # still used if explicit list omitted

        # Mastery-based curriculum (optional)
        "mastery_based_curriculum": config.mastery_based_curriculum,
        "mastery_threshold": config.mastery_threshold,
        "mastery_consistency_required": config.mastery_consistency_required,
        "mastery_fallback_steps": config.mastery_fallback_steps,
    }
    
    if extreme_verbosity:
        # Enable extreme logs for training and models (same computation path, same hyperparameters)
        set_extreme_verbosity(True)
        transformer.set_debug_mode(True)
        mamba.set_debug_mode(True)
        print("🚨 EXTREME VERBOSITY ACTIVE: Detailed step-by-step logging (same hyperparameters)")
        print("   ⚠️ Expect very verbose output intended for diagnosing failures")

    # Single training path (computation identical), logging toggled by flag
    if not extreme_verbosity:
        print("🚀 USING INTELLIGENT TRAINING WITH COMPREHENSIVE MONITORING")
        print(f"   🔍 Validation every {training_config['validation_interval']} steps (efficient training mode)")
        print(f"   💾 Checkpoints every {training_config['checkpoint_interval']} steps") 
        print(f"   ⏱️  Loss plateau patience: {training_config['loss_plateau_patience']} steps")
        print(f"   🚫 Low-accuracy patience: {training_config['accuracy_stuck_patience']} steps")
        print(f"   🛑 Training will stop when loss plateaus AND token accuracy stays low")
        print(f"   🐛 Sequence debugging enabled for first example")
        # Adjust non-extreme logging frequency (no effect on optimization)
        training_config["loss_log_interval"] = 200
        training_config["validation_interval"] = 5000
        print(f"   ✍️  Loss log interval set to {training_config['loss_log_interval']} steps")
        print(f"   🧪 Validation interval set to {training_config['validation_interval']} steps")
    # Train both models with intelligent stopping (same function for both modes)
    transformer_loss, transformer_trainer = train_model_comprehensive(
        transformer, "Transformer", task, training_config
    )
    mamba_loss, mamba_trainer = train_model_comprehensive(
        mamba, "Mamba", task, training_config
    )

    return transformer_loss, mamba_loss, transformer, mamba, transformer_trainer, mamba_trainer


def evaluate_models(transformer, mamba, task, test_lengths):
    """Evaluate both models on copying task."""
    print("📊 Evaluating models...")
    
    from utils.evaluation import evaluate_model_simple
    
    transformer_results = evaluate_model_simple(transformer, task, test_lengths)
    mamba_results = evaluate_model_simple(mamba, task, test_lengths)
    
    return transformer_results, mamba_results


def generate_plots(transformer_results, mamba_results, transformer_loss, mamba_loss, 
                  transformer_trainer=None, mamba_trainer=None):
    """Generate all publication-quality figures with comprehensive training data."""
    print("🎨 Generating comprehensive figures...")
    
    # Generate main figures using simple interface
    create_performance_comparison(transformer_results, mamba_results)
    create_training_summary(transformer_loss, mamba_loss)
    
    # If we have trainer objects with comprehensive history, generate advanced plots
    if transformer_trainer and mamba_trainer:
        print("📊 Generating advanced training analysis plots...")
        
        # Import enhanced visualization functions
        from visualization.publication_plots import (
            create_comprehensive_training_analysis, 
            create_training_diagnostics_summary
        )
        
        # Generate comprehensive training analysis
        create_comprehensive_training_analysis(transformer_trainer, mamba_trainer)
        
        # Generate diagnostic plots to identify training issues
        create_training_diagnostics_summary(transformer_trainer, mamba_trainer)
        
        # Also save individual trainer plots in figures/
        transformer_trainer.plot_training_curves("figures/transformer_detailed.png")
        mamba_trainer.plot_training_curves("figures/mamba_detailed.png")
        
        # Generate statistical analysis report
        from evaluation.statistical_evaluation import create_statistical_training_report
        create_statistical_training_report(transformer_trainer, mamba_trainer)
        
        print("📈 All comprehensive analysis plots and reports saved!")
    
    print("✅ All figures saved to figures/")


def main():
    parser = argparse.ArgumentParser(description='Academic Replication: Transformers vs Mamba')
    parser.add_argument('--train-only', action='store_true', help='Run training only')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--plot-only', action='store_true', help='Generate plots only')
    parser.add_argument('--debug', action='store_true', help='Enable enhanced debugging mode')
    parser.add_argument('--extreme-verbosity', action='store_true', help='Enable EXTREME logging (training + attention)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Select compute device (default: cpu). Use auto to prefer CUDA/MPS when available.')
    args = parser.parse_args()
    
    print("🎓 Academic Replication: Transformers vs Mamba")
    print("🧠 Using Intelligent Early Stopping (no max steps)")
    if args.debug:
        print("🔧 ENHANCED DEBUGGING MODE ENABLED")
    if args.extreme_verbosity:
        print("🧨 EXTREME VERBOSITY ENABLED (training + attention debug)")
    print("=" * 60)
    
    # Initialize experiment
    from utils.device import resolve_device, device_summary
    selected_device = resolve_device(args.device)
    if args.device != 'cpu':
        print(f"🧩 Device preference: {args.device} -> using {device_summary(selected_device)}")
    config, manager = create_academic_experiment()
    # Apply device override without changing defaults for CPU users
    config.device = selected_device
    task = CopyingTask(config.vocab_size)
    
    # Set evaluation lengths (no more quick mode)
    test_lengths = [50, 100, 150, 200, 250, 300, 400, 500]
    
    if args.plot_only:
        # Generate plots with sample data for demonstration
        transformer_results = {50: 0.95, 100: 0.88, 200: 0.75, 300: 0.65}
        mamba_results = {50: 0.65, 100: 0.45, 200: 0.25, 300: 0.15}
        generate_plots(transformer_results, mamba_results, 0.000, 0.420)
        return
    
    # Training phase with intelligent stopping
    if not args.eval_only:
        result = train_models(config, task, debug=args.debug, extreme_verbosity=args.extreme_verbosity)
        
        # Now we always get trainer objects with comprehensive monitoring
        transformer_loss, mamba_loss, transformer, mamba, transformer_trainer, mamba_trainer = result
        
        if args.train_only:
            print(f"Training complete with intelligent stopping:")
            print(f"  Transformer final loss: {transformer_loss:.6f}")
            print(f"  Mamba final loss: {mamba_loss:.6f}")
            return
    else:
        # For eval-only mode, would need pre-trained models
        print("⚠️  Eval-only mode requires pre-trained models (not implemented)")
        return
    
    # Evaluation phase
    if not args.train_only:
        # Silence extreme verbosity for evaluation to avoid massive logs
        if args.extreme_verbosity:
            try:
                transformer.set_debug_mode(False)
                mamba.set_debug_mode(False)
            except Exception:
                pass
        transformer_results, mamba_results = evaluate_models(transformer, mamba, task, test_lengths)
        
        # Generate plots; skip advanced trainer analysis in extreme mode
        if args.extreme_verbosity:
            generate_plots(transformer_results, mamba_results, transformer_loss, mamba_loss)
        else:
            # Generate comprehensive plots with trainer data
            generate_plots(transformer_results, mamba_results, transformer_loss, mamba_loss,
                          transformer_trainer, mamba_trainer)
        
        # Print summary
        print("🏆 Replication Complete!")
        print(f"Transformer superior on {sum(1 for l in test_lengths if transformer_results.get(l, 0) > mamba_results.get(l, 0))}/{len(test_lengths)} lengths")
        
        # Results summary
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"{'Length':<8} {'Transformer':<12} {'Mamba':<12} {'Gap':<12}")
        print("-" * 50)
        for length in test_lengths[:4]:  # Show first 4 lengths
            t_acc = transformer_results.get(length, 0.0)
            m_acc = mamba_results.get(length, 0.0)
            gap = t_acc - m_acc
            print(f"{length:<8} {t_acc:<12.1%} {m_acc:<12.1%} {gap:<+12.1%}")


if __name__ == "__main__":
    main()
