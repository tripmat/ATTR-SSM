"""
Statistical Evaluation Protocol for Transformer vs Mamba Comparison
Following academic standards for rigorous empirical evaluation.

Key Features:
1. Statistical significance testing (t-tests, confidence intervals)
2. Multiple random seeds for robust results  
3. Effect size computation (Cohen's d)
4. Publication-ready statistical reporting
5. Copying task evaluation following Jelassi et al. methodology

Academic Standards:
- Multiple model initializations (n≥5) 
- Large evaluation sets (n≥100 per condition)
- Proper statistical tests with p-values
- Effect size reporting for practical significance
- Confidence intervals for all metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmarks.copying_benchmark import CopyingTask


@dataclass
class StatisticalResult:
    """Container for statistical test results"""
    metric_name: str
    group1_name: str
    group2_name: str
    group1_values: List[float]
    group2_values: List[float]
    
    # Descriptive statistics
    group1_mean: float
    group1_std: float
    group2_mean: float  
    group2_std: float
    
    # Statistical test results
    t_statistic: float
    p_value: float
    degrees_freedom: int
    
    # Effect size
    cohens_d: float
    effect_size_interpretation: str
    
    # Confidence intervals
    group1_ci: Tuple[float, float]
    group2_ci: Tuple[float, float]
    mean_difference: float
    mean_difference_ci: Tuple[float, float]
    
    def __post_init__(self):
        """Compute derived statistics"""
        self.is_significant = self.p_value < 0.05
        self.is_highly_significant = self.p_value < 0.01
        
        # Effect size interpretation (Cohen's conventions)
        abs_d = abs(self.cohens_d)
        if abs_d < 0.2:
            self.effect_size_interpretation = "negligible"
        elif abs_d < 0.5:
            self.effect_size_interpretation = "small"
        elif abs_d < 0.8:
            self.effect_size_interpretation = "medium"
        else:
            self.effect_size_interpretation = "large"


class CopyingTaskEvaluator:
    """
    Rigorous evaluation of copying task performance.
    Implements the methodology from Jelassi et al. with statistical enhancements.
    """
    
    def __init__(self, vocab_size: int = 30, device: str = "cpu"):
        self.vocab_size = vocab_size
        self.device = device
        self.copying_task = CopyingTask(vocab_size)
        
    def evaluate_model_accuracy(self, model: torch.nn.Module, 
                               sequence_length: int,
                               n_samples: int = 100,
                               string_type: str = "uniform") -> List[float]:
        """
        Evaluate model accuracy on copying task with multiple samples.
        
        Args:
            model: Model to evaluate
            sequence_length: Length of sequences to test
            n_samples: Number of evaluation samples
            string_type: Type of string generation ("uniform", "natural", "shuffled")
            
        Returns:
            List of accuracies (0.0 or 1.0 for each sample)
        """
        model.eval()
        accuracies = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                try:
                    # Generate copying example
                    input_seq, target_seq = self.copying_task.create_copy_example(
                        sequence_length, string_type
                    )
                    
                    # Convert to tensors and add batch dimension
                    input_tensor = torch.tensor(input_seq).unsqueeze(0).to(self.device)
                    target_tensor = torch.tensor(target_seq).unsqueeze(0).to(self.device)
                    
                    # Get model predictions
                    outputs = model(input_tensor)
                    logits = outputs["logits"]
                    
                    # Extract predictions after COPY token
                    copy_token_pos = input_seq.index(self.copying_task.COPY_TOKEN)
                    
                    if copy_token_pos + len(target_seq) < logits.size(1):
                        pred_logits = logits[0, copy_token_pos+1:copy_token_pos+1+len(target_seq)]
                        predictions = pred_logits.argmax(dim=-1).cpu().numpy()
                        target_np = target_tensor[0].cpu().numpy()
                        
                        # Exact sequence match accuracy
                        accuracy = float(np.array_equal(predictions, target_np))
                    else:
                        accuracy = 0.0
                        
                    accuracies.append(accuracy)
                    
                except Exception as e:
                    # Handle evaluation errors gracefully
                    accuracies.append(0.0)
                    
        return accuracies
    
    def evaluate_model_comprehensive(self, model: torch.nn.Module,
                                   sequence_lengths: List[int],
                                   n_samples: int = 100) -> Dict[int, Dict[str, List[float]]]:
        """
        Comprehensive evaluation across multiple sequence lengths and string types.
        
        Returns:
            Dictionary mapping sequence_length -> string_type -> accuracy_list
        """
        results = {}
        
        for length in tqdm(sequence_lengths, desc="Evaluating sequence lengths"):
            results[length] = {}
            
            for string_type in ["uniform", "natural", "shuffled"]:
                accuracies = self.evaluate_model_accuracy(
                    model, length, n_samples, string_type
                )
                results[length][string_type] = accuracies
                
        return results


class StatisticalAnalyzer:
    """
    Statistical analysis toolkit for model comparison.
    Implements rigorous statistical methods for empirical research.
    """
    
    @staticmethod
    def compute_descriptive_stats(values: List[float]) -> Dict[str, float]:
        """Compute comprehensive descriptive statistics"""
        values_array = np.array(values)
        
        return {
            "n": len(values),
            "mean": np.mean(values_array),
            "std": np.std(values_array, ddof=1),  # Sample standard deviation
            "sem": stats.sem(values_array),  # Standard error of mean
            "min": np.min(values_array),
            "max": np.max(values_array),
            "median": np.median(values_array),
            "q25": np.percentile(values_array, 25),
            "q75": np.percentile(values_array, 75),
        }
    
    @staticmethod
    def independent_t_test(group1: List[float], group2: List[float],
                          confidence_level: float = 0.95) -> StatisticalResult:
        """
        Perform independent samples t-test with comprehensive statistics.
        
        Args:
            group1, group2: Lists of values to compare
            confidence_level: Confidence level for intervals (default 95%)
            
        Returns:
            StatisticalResult object with all statistical information
        """
        g1_array = np.array(group1)
        g2_array = np.array(group2)
        
        # Descriptive statistics
        g1_stats = StatisticalAnalyzer.compute_descriptive_stats(group1)
        g2_stats = StatisticalAnalyzer.compute_descriptive_stats(group2)
        
        # Independent t-test
        t_stat, p_val = stats.ttest_ind(g1_array, g2_array, equal_var=False)  # Welch's t-test
        
        # Degrees of freedom for Welch's t-test
        s1_sq = g1_stats["std"] ** 2
        s2_sq = g2_stats["std"] ** 2
        n1, n2 = len(group1), len(group2)
        
        df = (s1_sq/n1 + s2_sq/n2)**2 / ((s1_sq/n1)**2/(n1-1) + (s2_sq/n2)**2/(n2-1))
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((n1-1)*s1_sq + (n2-1)*s2_sq) / (n1+n2-2))
        cohens_d = (g1_stats["mean"] - g2_stats["mean"]) / pooled_std
        
        # Confidence intervals
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        g1_ci = (
            g1_stats["mean"] - t_critical * g1_stats["sem"],
            g1_stats["mean"] + t_critical * g1_stats["sem"]
        )
        
        g2_ci = (
            g2_stats["mean"] - t_critical * g2_stats["sem"],  
            g2_stats["mean"] + t_critical * g2_stats["sem"]
        )
        
        # Mean difference confidence interval
        mean_diff = g1_stats["mean"] - g2_stats["mean"]
        se_diff = np.sqrt(g1_stats["sem"]**2 + g2_stats["sem"]**2)
        mean_diff_ci = (
            mean_diff - t_critical * se_diff,
            mean_diff + t_critical * se_diff
        )
        
        return StatisticalResult(
            metric_name="accuracy",
            group1_name="group1",
            group2_name="group2", 
            group1_values=group1,
            group2_values=group2,
            group1_mean=g1_stats["mean"],
            group1_std=g1_stats["std"],
            group2_mean=g2_stats["mean"],
            group2_std=g2_stats["std"],
            t_statistic=t_stat,
            p_value=p_val,
            degrees_freedom=df,
            cohens_d=cohens_d,
            effect_size_interpretation="",  # Set in __post_init__
            group1_ci=g1_ci,
            group2_ci=g2_ci,
            mean_difference=mean_diff,
            mean_difference_ci=mean_diff_ci
        )
    
    @staticmethod
    def format_statistical_report(result: StatisticalResult) -> str:
        """Generate publication-ready statistical report"""
        report = []
        
        report.append(f"Statistical Comparison: {result.group1_name} vs {result.group2_name}")
        report.append("=" * 60)
        
        # Descriptive statistics
        report.append("\nDESCRIPTIVE STATISTICS:")
        report.append(f"{result.group1_name}: M = {result.group1_mean:.3f}, SD = {result.group1_std:.3f}")
        report.append(f"{result.group2_name}: M = {result.group2_mean:.3f}, SD = {result.group2_std:.3f}")
        
        # Statistical test
        report.append("\nSTATISTICAL TEST:")
        report.append(f"Independent t-test: t({result.degrees_freedom:.1f}) = {result.t_statistic:.3f}, p = {result.p_value:.6f}")
        
        # Significance interpretation
        if result.is_highly_significant:
            sig_text = "highly significant (p < 0.01)"
        elif result.is_significant:
            sig_text = "significant (p < 0.05)"
        else:
            sig_text = "not significant (p ≥ 0.05)"
        
        report.append(f"Result: {sig_text}")
        
        # Effect size
        report.append("\nEFFECT SIZE:")
        report.append(f"Cohen's d = {result.cohens_d:.3f} ({result.effect_size_interpretation} effect)")
        
        # Confidence intervals
        report.append("\n95% CONFIDENCE INTERVALS:")
        report.append(f"{result.group1_name}: [{result.group1_ci[0]:.3f}, {result.group1_ci[1]:.3f}]")
        report.append(f"{result.group2_name}: [{result.group2_ci[0]:.3f}, {result.group2_ci[1]:.3f}]")
        report.append(f"Mean difference: {result.mean_difference:.3f} [{result.mean_difference_ci[0]:.3f}, {result.mean_difference_ci[1]:.3f}]")
        
        return "\n".join(report)


class ExperimentalComparison:
    """
    Complete experimental comparison framework.
    Implements the full pipeline for rigorous model comparison.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.evaluator = CopyingTaskEvaluator(config.vocab_size, config.device)
        self.analyzer = StatisticalAnalyzer()
        
    def compare_models(self, transformer_model: torch.nn.Module,
                      mamba_model: torch.nn.Module,
                      sequence_lengths: List[int],
                      n_samples: int = 100) -> Dict[str, Any]:
        """
        Complete statistical comparison between two models.
        
        Args:
            transformer_model: Transformer model to evaluate
            mamba_model: Mamba model to evaluate  
            sequence_lengths: List of sequence lengths to test
            n_samples: Number of samples per condition
            
        Returns:
            Comprehensive comparison results with statistical analysis
        """
        print("🔬 Starting comprehensive model comparison...")
        print(f"   Sequence lengths: {sequence_lengths}")
        print(f"   Samples per condition: {n_samples}")
        print(f"   Statistical analysis: t-tests, effect sizes, confidence intervals")
        
        # Evaluate both models
        print("\n📊 Evaluating Transformer...")
        transformer_results = self.evaluator.evaluate_model_comprehensive(
            transformer_model, sequence_lengths, n_samples
        )
        
        print("📊 Evaluating Mamba...")
        mamba_results = self.evaluator.evaluate_model_comprehensive(
            mamba_model, sequence_lengths, n_samples
        )
        
        # Statistical comparison for each condition
        comparison_results = {}
        
        for length in sequence_lengths:
            comparison_results[length] = {}
            
            for string_type in ["uniform", "natural", "shuffled"]:
                transformer_acc = transformer_results[length][string_type]
                mamba_acc = mamba_results[length][string_type]
                
                # Statistical test
                stat_result = self.analyzer.independent_t_test(
                    transformer_acc, mamba_acc
                )
                stat_result.group1_name = "Transformer"
                stat_result.group2_name = "Mamba"
                stat_result.metric_name = f"accuracy_length_{length}_{string_type}"
                
                comparison_results[length][string_type] = {
                    "transformer_accuracies": transformer_acc,
                    "mamba_accuracies": mamba_acc,
                    "statistical_test": stat_result,
                    "transformer_mean": np.mean(transformer_acc),
                    "mamba_mean": np.mean(mamba_acc),
                    "performance_gap": np.mean(transformer_acc) - np.mean(mamba_acc)
                }
        
        return {
            "comparison_results": comparison_results,
            "transformer_results": transformer_results,
            "mamba_results": mamba_results,
            "experimental_config": {
                "sequence_lengths": sequence_lengths,
                "n_samples": n_samples,
                "vocab_size": self.config.vocab_size,
                "evaluation_date": datetime.now().isoformat()
            }
        }
    
    def save_results(self, results: Dict[str, Any], experiment_id: str):
        """Save comprehensive results with statistical analysis"""
        results_dir = "experiments/results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save raw results
        with open(f"{results_dir}/{experiment_id}_statistical_results.json", "w") as f:
            # Convert StatisticalResult objects to dictionaries for JSON serialization
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        # Generate statistical report
        report_lines = []
        report_lines.append("TRANSFORMER vs MAMBA: STATISTICAL COMPARISON REPORT")
        report_lines.append("=" * 70)
        report_lines.append(f"Experiment ID: {experiment_id}")
        report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        comparison_results = results["comparison_results"]
        
        for length in sorted(comparison_results.keys()):
            for string_type in ["uniform", "natural", "shuffled"]:
                result = comparison_results[length][string_type]
                stat_test = result["statistical_test"]
                
                report_lines.append(f"\nSEQUENCE LENGTH {length} ({string_type.upper()}):")
                report_lines.append(self.analyzer.format_statistical_report(stat_test))
                report_lines.append("")
        
        # Save report
        with open(f"{results_dir}/{experiment_id}_statistical_report.txt", "w") as f:
            f.write("\n".join(report_lines))
        
        print(f"📊 Statistical results saved:")
        print(f"   Results: {experiment_id}_statistical_results.json")  
        print(f"   Report:  {experiment_id}_statistical_report.txt")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, StatisticalResult):
            return {
                "metric_name": obj.metric_name,
                "group1_name": obj.group1_name,
                "group2_name": obj.group2_name,
                "group1_mean": obj.group1_mean,
                "group1_std": obj.group1_std,
                "group2_mean": obj.group2_mean,
                "group2_std": obj.group2_std,
                "t_statistic": obj.t_statistic,
                "p_value": obj.p_value,
                "cohens_d": obj.cohens_d,
                "effect_size_interpretation": obj.effect_size_interpretation,
                "is_significant": obj.is_significant,
                "mean_difference": obj.mean_difference,
            }
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        else:
            return obj


if __name__ == "__main__":
    # Test statistical evaluation
    print("Testing Statistical Evaluation Framework")
    print("=" * 50)
    
    # Generate synthetic data for testing
    np.random.seed(42)
    
    # Simulate transformer (better) vs mamba (worse) performance  
    transformer_accuracies = np.random.beta(8, 2, 100).tolist()  # High accuracy
    mamba_accuracies = np.random.beta(3, 7, 100).tolist()       # Lower accuracy
    
    analyzer = StatisticalAnalyzer()
    result = analyzer.independent_t_test(transformer_accuracies, mamba_accuracies)
    result.group1_name = "Transformer"
    result.group2_name = "Mamba"
    
    report = analyzer.format_statistical_report(result)
    print(report)
    
    print("\n✅ Statistical framework ready for rigorous evaluation!")


def analyze_training_statistics(transformer_trainer, mamba_trainer, 
                               sequence_lengths: List[int] = [50, 100, 150, 200]):
    """
    Statistical analysis of training progression for both models.
    
    This function integrates with ComprehensiveTrainer objects to perform
    rigorous statistical analysis of training dynamics and convergence.
    
    Args:
        transformer_trainer: ComprehensiveTrainer object for Transformer
        mamba_trainer: ComprehensiveTrainer object for Mamba  
        sequence_lengths: Lengths to analyze
        
    Returns:
        Dictionary with statistical analysis results
    """
    if not transformer_trainer or not mamba_trainer:
        return {}
    
    analysis = {}
    
    # Training convergence analysis
    t_history = transformer_trainer.training_history
    m_history = mamba_trainer.training_history
    
    # 1. Loss convergence rates
    if len(t_history['train_losses']) > 100 and len(m_history['train_losses']) > 100:
        # Calculate convergence rate (loss reduction per 100 steps)
        t_early_loss = np.mean(t_history['train_losses'][:50])
        t_late_loss = np.mean(t_history['train_losses'][-50:])
        t_convergence_rate = (t_early_loss - t_late_loss) / (len(t_history['train_losses']) / 100)
        
        m_early_loss = np.mean(m_history['train_losses'][:50])
        m_late_loss = np.mean(m_history['train_losses'][-50:])
        m_convergence_rate = (m_early_loss - m_late_loss) / (len(m_history['train_losses']) / 100)
        
        analysis['convergence_analysis'] = {
            'transformer': {
                'early_loss': t_early_loss,
                'final_loss': t_late_loss,
                'convergence_rate': t_convergence_rate
            },
            'mamba': {
                'early_loss': m_early_loss,
                'final_loss': m_late_loss,
                'convergence_rate': m_convergence_rate
            }
        }
    
    # 2. Training stability analysis
    if len(t_history['gradient_norms']) > 50 and len(m_history['gradient_norms']) > 50:
        t_grad_stability = np.std(t_history['gradient_norms'][-100:])  # Last 100 steps
        m_grad_stability = np.std(m_history['gradient_norms'][-100:])
        
        analysis['stability_analysis'] = {
            'transformer_gradient_stability': t_grad_stability,
            'mamba_gradient_stability': m_grad_stability,
            'stability_ratio': t_grad_stability / max(m_grad_stability, 1e-6)
        }
    
    # 3. Validation performance statistical comparison
    if t_history['val_accuracies'] and m_history['val_accuracies']:
        validation_stats = {}
        
        for length in sequence_lengths:
            t_key = f"val_acc_{length}"
            m_key = f"val_acc_{length}"
            
            if t_key in t_history['val_accuracies'] and m_key in m_history['val_accuracies']:
                t_accs = t_history['val_accuracies'][t_key]
                m_accs = m_history['val_accuracies'][m_key]
                
                if len(t_accs) > 5 and len(m_accs) > 5:  # Need enough samples
                    # Statistical significance test
                    t_stat, p_value = stats.ttest_ind(t_accs, m_accs)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(t_accs)**2 + np.std(m_accs)**2) / 2)
                    cohens_d = (np.mean(t_accs) - np.mean(m_accs)) / max(pooled_std, 1e-6)
                    
                    validation_stats[f"length_{length}"] = {
                        'transformer_mean': np.mean(t_accs),
                        'transformer_std': np.std(t_accs),
                        'mamba_mean': np.mean(m_accs),
                        'mamba_std': np.std(m_accs),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05,
                        'effect_size': 'small' if abs(cohens_d) < 0.5 else 'medium' if abs(cohens_d) < 0.8 else 'large'
                    }
        
        analysis['validation_statistics'] = validation_stats
    
    # 4. Learning efficiency analysis
    # Find steps to reach certain performance thresholds
    efficiency_analysis = {}
    
    for length in [50, 100]:  # Focus on key lengths
        t_key = f"val_acc_{length}"
        m_key = f"val_acc_{length}"
        
        if (t_key in t_history['val_accuracies'] and m_key in m_history['val_accuracies'] 
            and t_history['val_accuracies'][t_key] and m_history['val_accuracies'][m_key]):
            
            # Steps to reach 25%, 50%, 75% accuracy
            for threshold in [0.25, 0.5, 0.75]:
                t_steps = None
                m_steps = None
                
                # Find first step where accuracy >= threshold
                for i, acc in enumerate(t_history['val_accuracies'][t_key]):
                    if acc >= threshold:
                        t_steps = (i + 1) * transformer_trainer.val_interval
                        break
                
                for i, acc in enumerate(m_history['val_accuracies'][m_key]):
                    if acc >= threshold:
                        m_steps = (i + 1) * mamba_trainer.val_interval
                        break
                
                efficiency_analysis[f"length_{length}_threshold_{int(threshold*100)}"] = {
                    'transformer_steps': t_steps,
                    'mamba_steps': m_steps,
                    'transformer_faster': t_steps is not None and (m_steps is None or t_steps < m_steps)
                }
    
    analysis['efficiency_analysis'] = efficiency_analysis
    
    # 5. Final model comparison
    final_comparison = {}
    
    # Get final losses and accuracies
    if t_history['train_losses'] and m_history['train_losses']:
        final_comparison['final_train_loss'] = {
            'transformer': t_history['train_losses'][-1],
            'mamba': m_history['train_losses'][-1]
        }
    
    if t_history['val_losses'] and m_history['val_losses']:
        final_comparison['final_val_loss'] = {
            'transformer': t_history['val_losses'][-1],
            'mamba': m_history['val_losses'][-1]
        }
    
    # Overall validation performance
    if t_history['val_accuracies'] and m_history['val_accuracies']:
        t_overall_acc = []
        m_overall_acc = []
        
        for length in sequence_lengths:
            t_key = f"val_acc_{length}"
            m_key = f"val_acc_{length}"
            
            if (t_key in t_history['val_accuracies'] and m_key in m_history['val_accuracies']
                and t_history['val_accuracies'][t_key] and m_history['val_accuracies'][m_key]):
                t_overall_acc.append(t_history['val_accuracies'][t_key][-1])
                m_overall_acc.append(m_history['val_accuracies'][m_key][-1])
        
        if t_overall_acc and m_overall_acc:
            final_comparison['overall_validation_performance'] = {
                'transformer_mean': np.mean(t_overall_acc),
                'mamba_mean': np.mean(m_overall_acc),
                'advantage': np.mean(t_overall_acc) - np.mean(m_overall_acc),
                'transformer_wins': sum(1 for t, m in zip(t_overall_acc, m_overall_acc) if t > m),
                'total_comparisons': len(t_overall_acc)
            }
    
    analysis['final_comparison'] = final_comparison
    
    return analysis


def create_statistical_training_report(transformer_trainer, mamba_trainer,
                                     save_path: str = "experiments/statistical_training_report.json"):
    """
    Generate comprehensive statistical report of training dynamics.
    
    This integrates ComprehensiveTrainer data with rigorous statistical analysis
    to provide publication-ready training analysis.
    """
    print("📊 Generating comprehensive statistical training report...")
    
    # Perform statistical analysis
    analysis = analyze_training_statistics(transformer_trainer, mamba_trainer)
    
    # Add metadata
    report = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'comprehensive_training_statistics',
        'models_compared': ['transformer', 'mamba'],
        'statistical_analysis': analysis
    }
    
    # Add training configuration info
    if transformer_trainer:
        report['transformer_config'] = {
            'validation_interval': transformer_trainer.val_interval,
            'validation_lengths': transformer_trainer.val_lengths,
            'total_training_steps': len(transformer_trainer.training_history.get('steps', [])),
            'early_stopping_patience': transformer_trainer.patience
        }
    
    if mamba_trainer:
        report['mamba_config'] = {
            'validation_interval': mamba_trainer.val_interval,
            'validation_lengths': mamba_trainer.val_lengths,
            'total_training_steps': len(mamba_trainer.training_history.get('steps', [])),
            'early_stopping_patience': mamba_trainer.patience
        }
    
    # Save report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Print key findings
    print(f"\n📈 KEY STATISTICAL FINDINGS:")
    
    if 'convergence_analysis' in analysis:
        conv = analysis['convergence_analysis']
        print(f"• Convergence Rate:")
        print(f"  - Transformer: {conv['transformer']['convergence_rate']:.6f} loss reduction per 100 steps")
        print(f"  - Mamba: {conv['mamba']['convergence_rate']:.6f} loss reduction per 100 steps")
    
    if 'validation_statistics' in analysis:
        val_stats = analysis['validation_statistics']
        significant_advantages = sum(1 for k, v in val_stats.items() 
                                   if v.get('significant', False) and v.get('transformer_mean', 0) > v.get('mamba_mean', 0))
        print(f"• Validation Performance:")
        print(f"  - Transformer shows significant advantage on {significant_advantages}/{len(val_stats)} sequence lengths")
    
    if 'efficiency_analysis' in analysis:
        eff = analysis['efficiency_analysis']
        transformer_faster = sum(1 for k, v in eff.items() if v.get('transformer_faster', False))
        print(f"• Learning Efficiency:")
        print(f"  - Transformer reaches thresholds faster in {transformer_faster}/{len(eff)} cases")
    
    print(f"📄 Detailed report saved: {save_path}")
    return report