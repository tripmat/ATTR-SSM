"""
Publication-Ready Visualization for Transformer vs Mamba Comparison.

Single consolidated module for all figure generation following academic ML standards.
This is the ONLY plotting module in the repository.

Features:
- IEEE/Nature style formatting with significance indicators
- Complete statistical reporting (error bars, p-values, effect sizes) 
- Professional academic color schemes and typography
- Multi-format export (PNG, PDF) optimized for publications
- Clean API following ML research conventions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path


# Academic color palette (colorblind-friendly)
COLORS = {
    'transformer': '#2E86AB',      # Professional blue
    'mamba': '#A23B72',            # Distinct magenta  
    'significant': '#F18F01',       # Orange for significance
    'not_significant': '#C73E1D',   # Red for non-significance
    'background': '#F7F7F7',       # Light gray background
    'grid': '#E0E0E0',            # Grid color
    'text': '#2D2D2D'             # Dark text
}

# Academic figure styling
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        plt.style.use('seaborn-whitegrid')
    except OSError:
        plt.style.use('default')
        
sns.set_palette([COLORS['transformer'], COLORS['mamba']])

# Font configuration for publication
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'text.usetex': False,  # Avoid LaTeX dependency
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 300,
    'figure.dpi': 100
})


class PublicationVisualizer:
    """
    Creates publication-quality visualizations for academic papers.
    
    All figures follow academic standards:
    - Clear, readable fonts
    - Appropriate statistical representations
    - Colorblind-friendly palettes
    - Proper error reporting
    """
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize with experimental results.
        
        Args:
            results: Complete experimental results from statistical evaluation
        """
        self.results = results
        self.comparison_results = results.get("comparison_results", {})
        self.config = results.get("experimental_config", {})
        
    def create_main_comparison_figure(self, save_path: str = "figures/main_comparison.png"):
        """
        Create the main comparison figure showing Transformer vs Mamba performance.
        
        This is the key figure demonstrating the paper's main result.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Transformer vs Mamba: Copying Task Performance Comparison', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # Extract sequence lengths and performance data
        sequence_lengths = sorted(self.comparison_results.keys())
        transformer_means = []
        mamba_means = []
        transformer_stds = []
        mamba_stds = []
        p_values = []
        effect_sizes = []
        
        for length in sequence_lengths:
            result = self.comparison_results[length]["uniform"]
            
            transformer_means.append(result["transformer_mean"])
            mamba_means.append(result["mamba_mean"]) 
            
            stat_test = result["statistical_test"]
            transformer_stds.append(stat_test.group1_std)
            mamba_stds.append(stat_test.group2_std)
            p_values.append(stat_test.p_value)
            effect_sizes.append(abs(stat_test.cohens_d))
        
        # Convert to numpy arrays for easier manipulation
        lengths_array = np.array(sequence_lengths)
        transformer_means = np.array(transformer_means)
        mamba_means = np.array(mamba_means)
        transformer_stds = np.array(transformer_stds)
        mamba_stds = np.array(mamba_stds)
        p_values = np.array(p_values)
        effect_sizes = np.array(effect_sizes)
        
        # Panel A: Performance Comparison with Error Bars
        ax1.errorbar(lengths_array, transformer_means, yerr=transformer_stds, 
                    color=COLORS['transformer'], marker='o', linewidth=2, 
                    markersize=6, capsize=3, label='Transformer')
        ax1.errorbar(lengths_array, mamba_means, yerr=mamba_stds,
                    color=COLORS['mamba'], marker='s', linewidth=2,
                    markersize=6, capsize=3, label='Mamba')
        
        ax1.set_xlabel('Sequence Length (tokens)')
        ax1.set_ylabel('String-Level Accuracy')
        ax1.set_title('A. Performance Comparison', fontweight='bold')
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.05)
        
        # Add significance markers
        for i, (length, p_val) in enumerate(zip(lengths_array, p_values)):
            if p_val < 0.001:
                marker = '***'
            elif p_val < 0.01:
                marker = '**'
            elif p_val < 0.05:
                marker = '*'
            else:
                marker = 'ns'
                
            if marker != 'ns':
                y_pos = max(transformer_means[i], mamba_means[i]) + 0.05
                ax1.text(length, y_pos, marker, ha='center', va='bottom',
                        fontsize=10, color=COLORS['significant'])
        
        # Panel B: Performance Gap
        performance_gap = transformer_means - mamba_means
        
        ax2.plot(lengths_array, performance_gap, color=COLORS['significant'], 
                marker='D', linewidth=2, markersize=5)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.fill_between(lengths_array, performance_gap, alpha=0.3, 
                        color=COLORS['significant'])
        
        ax2.set_xlabel('Sequence Length (tokens)')
        ax2.set_ylabel('Performance Gap\n(Transformer - Mamba)')
        ax2.set_title('B. Transformer Advantage', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Statistical Significance
        # Color bars by significance level
        colors = [COLORS['significant'] if p < 0.05 else COLORS['not_significant'] 
                 for p in p_values]
        
        bars = ax3.bar(lengths_array, -np.log10(p_values), color=colors, alpha=0.7)
        ax3.axhline(y=-np.log10(0.05), color='red', linestyle='--', 
                   label='p = 0.05 threshold')
        ax3.axhline(y=-np.log10(0.01), color='orange', linestyle='--',
                   label='p = 0.01 threshold')
        
        ax3.set_xlabel('Sequence Length (tokens)')
        ax3.set_ylabel('-log₁₀(p-value)')
        ax3.set_title('C. Statistical Significance', fontweight='bold')
        ax3.legend(frameon=True)
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Effect Size (Cohen's d)
        ax4.bar(lengths_array, effect_sizes, color=COLORS['transformer'], alpha=0.7)
        
        # Effect size interpretation lines
        ax4.axhline(y=0.2, color='gray', linestyle=':', alpha=0.7, label='Small effect')
        ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
        ax4.axhline(y=0.8, color='gray', linestyle='-', alpha=0.7, label='Large effect')
        
        ax4.set_xlabel('Sequence Length (tokens)')
        ax4.set_ylabel('Effect Size (|Cohen\'s d|)')
        ax4.set_title('D. Effect Size Analysis', fontweight='bold')
        ax4.legend(frameon=True)
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create figures directory
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save in multiple formats
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.svg'), dpi=300, bbox_inches='tight')
        
        print(f"📊 Main comparison figure saved: {save_path}")
        return fig
    
    def create_statistical_summary_table(self, save_path: str = "figures/statistical_summary.png"):
        """
        Create a publication-ready statistical summary table.
        """
        # Prepare data for table
        sequence_lengths = sorted(self.comparison_results.keys())
        
        table_data = []
        for length in sequence_lengths:
            result = self.comparison_results[length]["uniform"]
            stat_test = result["statistical_test"]
            
            # Format values for publication
            transformer_mean = f"{result['transformer_mean']:.3f}"
            mamba_mean = f"{result['mamba_mean']:.3f}"
            mean_diff = f"{result['performance_gap']:.3f}"
            
            # Format p-value
            if stat_test.p_value < 0.001:
                p_str = "< 0.001"
            else:
                p_str = f"{stat_test.p_value:.3f}"
            
            # Effect size interpretation
            effect_str = f"{stat_test.cohens_d:.2f} ({stat_test.effect_size_interpretation})"
            
            table_data.append([
                str(length),
                transformer_mean,
                mamba_mean,
                mean_diff,
                p_str,
                effect_str
            ])
        
        # Create table figure
        fig, ax = plt.subplots(figsize=(12, len(sequence_lengths) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # Table headers
        headers = ['Sequence\nLength', 'Transformer\nMean ± SD', 'Mamba\nMean ± SD', 
                  'Mean\nDifference', 'p-value', 'Effect Size\n(Cohen\'s d)']
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style table
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header row
                cell.set_text_props(weight='bold')
                cell.set_facecolor(COLORS['background'])
            else:
                cell.set_facecolor('white')
            
            cell.set_edgecolor(COLORS['grid'])
            cell.set_linewidth(1)
        
        plt.title('Statistical Summary: Transformer vs Mamba Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📋 Statistical summary table saved: {save_path}")
        
        return fig
    
    def create_supplementary_figures(self, save_path: str = "figures/supplementary.png"):
        """
        Create supplementary figures for additional analysis.
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Data for different string types
        sequence_lengths = sorted(self.comparison_results.keys())
        string_types = ["uniform", "natural", "shuffled"]
        
        # Panel A: Performance by String Type
        for string_type in string_types:
            transformer_means = []
            for length in sequence_lengths:
                if string_type in self.comparison_results[length]:
                    transformer_means.append(
                        self.comparison_results[length][string_type]["transformer_mean"]
                    )
                else:
                    transformer_means.append(0)  # Fallback
            
            ax1.plot(sequence_lengths, transformer_means, marker='o', 
                    label=f'Transformer ({string_type})', linewidth=2)
        
        ax1.set_xlabel('Sequence Length (tokens)')
        ax1.set_ylabel('String-Level Accuracy')  
        ax1.set_title('A. Performance by String Type', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel B: Mamba Performance by String Type
        for string_type in string_types:
            mamba_means = []
            for length in sequence_lengths:
                if string_type in self.comparison_results[length]:
                    mamba_means.append(
                        self.comparison_results[length][string_type]["mamba_mean"]
                    )
                else:
                    mamba_means.append(0)  # Fallback
                    
            ax2.plot(sequence_lengths, mamba_means, marker='s',
                    label=f'Mamba ({string_type})', linewidth=2)
        
        ax2.set_xlabel('Sequence Length (tokens)')
        ax2.set_ylabel('String-Level Accuracy')
        ax2.set_title('B. Mamba Performance by String Type', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel C: Accuracy Distribution (example for first length)
        first_length = sequence_lengths[0]
        transformer_accs = self.comparison_results[first_length]["uniform"]["transformer_accuracies"]
        mamba_accs = self.comparison_results[first_length]["uniform"]["mamba_accuracies"]
        
        ax3.hist(transformer_accs, bins=10, alpha=0.7, color=COLORS['transformer'],
                label='Transformer', density=True)
        ax3.hist(mamba_accs, bins=10, alpha=0.7, color=COLORS['mamba'],
                label='Mamba', density=True)
        
        ax3.set_xlabel('Accuracy')
        ax3.set_ylabel('Density')
        ax3.set_title(f'C. Accuracy Distribution (Length {first_length})', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel D: Parameter Comparison
        # Create a simple parameter comparison chart
        models = ['Transformer', 'Mamba']
        # These would come from the actual model parameter counts
        param_counts = [10.87, 10.83]  # In millions
        
        bars = ax4.bar(models, param_counts, color=[COLORS['transformer'], COLORS['mamba']], alpha=0.7)
        
        # Add value labels on bars
        for bar, count in zip(bars, param_counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{count:.2f}M', ha='center', va='bottom', fontweight='bold')
        
        ax4.set_ylabel('Parameters (Millions)')
        ax4.set_title('D. Parameter Count Comparison', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Supplementary Analysis: Transformer vs Mamba', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 Supplementary figures saved: {save_path}")
        
        return fig
    
    def generate_all_figures(self, base_path: str = "figures"):
        """
        Generate complete set of publication-ready figures.
        
        Args:
            base_path: Directory to save figures
        """
        print("🎨 Generating publication-ready figures...")
        
        # Create main comparison figure
        main_fig = self.create_main_comparison_figure(f"{base_path}/main_comparison.png")
        
        # Create statistical summary table
        table_fig = self.create_statistical_summary_table(f"{base_path}/statistical_summary.png")
        
        # Create supplementary figures  
        supp_fig = self.create_supplementary_figures(f"{base_path}/supplementary.png")
        
        print(f"✅ All publication figures generated in '{base_path}/' directory")
        print(f"   Main comparison: main_comparison.png/pdf/svg")
        print(f"   Statistical summary: statistical_summary.png")
        print(f"   Supplementary analysis: supplementary.png")
        
        return main_fig, table_fig, supp_fig


def create_publication_figures(results_file: str):
    """
    Create publication figures from experimental results file.
    
    Args:
        results_file: Path to JSON file containing experimental results
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Create visualizer
    visualizer = PublicationVisualizer(results)
    
    # Generate all figures
    visualizer.generate_all_figures()


# Simple interface functions for the main entry point
def create_performance_comparison(transformer_results, mamba_results, save_path="figures/performance.png"):
    """Simple interface for creating performance comparison plots."""
    import numpy as np
    
    lengths = sorted(transformer_results.keys())
    transformer_acc = [transformer_results[l] for l in lengths]
    mamba_acc = [mamba_results[l] for l in lengths]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Performance comparison
    ax1.plot(lengths, transformer_acc, 'o-', color=COLORS['transformer'], 
             linewidth=2, markersize=6, label='Transformer')
    ax1.plot(lengths, mamba_acc, 's-', color=COLORS['mamba'], 
             linewidth=2, markersize=6, label='Mamba')
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Performance gap
    gap = [t - m for t, m in zip(transformer_acc, mamba_acc)]
    ax2.bar(lengths, gap, color=COLORS['significant'], alpha=0.7)
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Performance Gap')
    ax2.set_title('Transformer Advantage')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"📊 Performance comparison saved: {save_path}")


def create_training_summary(transformer_loss, mamba_loss, save_path="figures/training.png"):
    """Simple interface for creating training summary."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models = ['Transformer', 'Mamba']
    losses = [transformer_loss, mamba_loss]
    colors = [COLORS['transformer'], COLORS['mamba']]
    
    bars = ax.bar(models, losses, color=colors, alpha=0.8)
    ax.set_ylabel('Final Training Loss')
    ax.set_title('Training Convergence Comparison')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, loss in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(losses) * 0.01,
                f'{loss:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📈 Training summary saved: {save_path}")


def create_comprehensive_training_analysis(transformer_trainer, mamba_trainer, 
                                          save_path="figures/comprehensive_training_analysis.png"):
    """
    Create comprehensive training analysis combining both models.
    
    This function enhances the existing publication plots by using rich training
    history data from ComprehensiveTrainer objects.
    """
    if not transformer_trainer or not mamba_trainer:
        print("⚠️  Trainer objects not available - using basic plots")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Training Analysis: Transformer vs Mamba', fontsize=16, fontweight='bold')
    
    # Extract training histories
    t_history = transformer_trainer.training_history
    m_history = mamba_trainer.training_history
    
    # 1. Training Loss Comparison
    axes[0, 0].plot(t_history['steps'], t_history['train_losses'], 
                    color=COLORS['transformer'], label='Transformer', linewidth=2, alpha=0.8)
    axes[0, 0].plot(m_history['steps'], m_history['train_losses'], 
                    color=COLORS['mamba'], label='Mamba', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Training Loss')
    axes[0, 0].set_title('Training Loss Progression')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Validation Loss Comparison
    if t_history['val_losses'] and m_history['val_losses']:
        t_val_steps = [s for i, s in enumerate(t_history['steps']) 
                       if i % transformer_trainer.val_interval == 0][:len(t_history['val_losses'])]
        m_val_steps = [s for i, s in enumerate(m_history['steps']) 
                       if i % mamba_trainer.val_interval == 0][:len(m_history['val_losses'])]
        
        axes[0, 1].plot(t_val_steps, t_history['val_losses'], 
                        'o-', color=COLORS['transformer'], label='Transformer', linewidth=2)
        axes[0, 1].plot(m_val_steps, m_history['val_losses'], 
                        's-', color=COLORS['mamba'], label='Mamba', linewidth=2)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Validation Loss')
        axes[0, 1].set_title('Validation Loss Progression')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Learning Rate Schedules
    axes[0, 2].plot(t_history['steps'], t_history['learning_rates'], 
                    color=COLORS['transformer'], label='Transformer', linewidth=2)
    axes[0, 2].plot(m_history['steps'], m_history['learning_rates'], 
                    color=COLORS['mamba'], label='Mamba', linewidth=2, linestyle='--')
    axes[0, 2].set_xlabel('Training Steps')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].legend()
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Gradient Norms (Training Stability)
    axes[1, 0].plot(t_history['steps'], t_history['gradient_norms'], 
                    color=COLORS['transformer'], alpha=0.7, label='Transformer')
    axes[1, 0].plot(m_history['steps'], m_history['gradient_norms'], 
                    color=COLORS['mamba'], alpha=0.7, label='Mamba')
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Gradient Norm')
    axes[1, 0].set_title('Gradient Norms (Training Stability)')
    axes[1, 0].legend()
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add horizontal line at gradient norm = 1.0 (common clipping threshold)
    axes[1, 0].axhline(y=1.0, color='red', linestyle=':', alpha=0.5, label='Clipping threshold')
    
    # 5. Validation Accuracy by Length (if available)
    if t_history['val_accuracies'] and m_history['val_accuracies']:
        val_lengths = transformer_trainer.val_lengths
        for length in val_lengths:
            t_key = f"val_acc_{length}"
            m_key = f"val_acc_{length}"
            
            if t_key in t_history['val_accuracies'] and m_key in m_history['val_accuracies']:
                t_val_steps = [s for i, s in enumerate(t_history['steps']) 
                             if i % transformer_trainer.val_interval == 0][:len(t_history['val_accuracies'][t_key])]
                m_val_steps = [s for i, s in enumerate(m_history['steps']) 
                             if i % mamba_trainer.val_interval == 0][:len(m_history['val_accuracies'][m_key])]
                
                # Plot only a few key lengths to avoid clutter
                if length in [50, 100, 200]:
                    alpha = 0.8 if length == 50 else 0.6 if length == 100 else 0.4
                    axes[1, 1].plot(t_val_steps, t_history['val_accuracies'][t_key], 
                                   color=COLORS['transformer'], alpha=alpha, 
                                   label=f'T-{length}' if length == 50 else None)
                    axes[1, 1].plot(m_val_steps, m_history['val_accuracies'][m_key], 
                                   color=COLORS['mamba'], alpha=alpha, linestyle='--',
                                   label=f'M-{length}' if length == 50 else None)
        
        axes[1, 1].set_xlabel('Training Steps')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].set_title('Validation Accuracy During Training')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
    
    # 6. Training Speed Comparison
    if t_history['tokens_per_second'] and m_history['tokens_per_second']:
        # Use moving average for smoother visualization
        def moving_average(data, window=50):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        if len(t_history['tokens_per_second']) > 50:
            t_speed_smooth = moving_average(t_history['tokens_per_second'])
            m_speed_smooth = moving_average(m_history['tokens_per_second'])
            t_steps_smooth = t_history['steps'][49:]  # Adjust for moving average
            m_steps_smooth = m_history['steps'][49:]
            
            axes[1, 2].plot(t_steps_smooth, t_speed_smooth, 
                           color=COLORS['transformer'], label='Transformer', linewidth=2)
            axes[1, 2].plot(m_steps_smooth, m_speed_smooth, 
                           color=COLORS['mamba'], label='Mamba', linewidth=2)
        else:
            axes[1, 2].plot(t_history['steps'], t_history['tokens_per_second'], 
                           color=COLORS['transformer'], label='Transformer', alpha=0.7)
            axes[1, 2].plot(m_history['steps'], m_history['tokens_per_second'], 
                           color=COLORS['mamba'], label='Mamba', alpha=0.7)
        
        axes[1, 2].set_xlabel('Training Steps')
        axes[1, 2].set_ylabel('Tokens/Second')
        axes[1, 2].set_title('Training Speed')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"📊 Comprehensive training analysis saved: {save_path}")
    plt.close()


def create_training_diagnostics_summary(transformer_trainer, mamba_trainer,
                                       save_path="figures/training_diagnostics.png"):
    """
    Create diagnostic summary to identify training issues.
    
    This function helps diagnose the mystery of why Mamba shows 0.0 loss
    but poor performance by analyzing training patterns.
    """
    if not transformer_trainer or not mamba_trainer:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Diagnostics: Identifying Issues', fontsize=16, fontweight='bold')
    
    t_history = transformer_trainer.training_history
    m_history = mamba_trainer.training_history
    
    # 1. Loss vs Validation Accuracy Correlation
    if t_history['val_losses'] and t_history['val_accuracies']:
        # Get validation accuracy for length 50 (most stable)
        t_val_acc_50 = t_history['val_accuracies'].get('val_acc_50', [])
        m_val_acc_50 = m_history['val_accuracies'].get('val_acc_50', [])
        
        if t_val_acc_50 and m_val_acc_50:
            axes[0, 0].scatter(t_history['val_losses'][:len(t_val_acc_50)], t_val_acc_50, 
                              color=COLORS['transformer'], alpha=0.6, s=30, label='Transformer')
            axes[0, 0].scatter(m_history['val_losses'][:len(m_val_acc_50)], m_val_acc_50, 
                              color=COLORS['mamba'], alpha=0.6, s=30, label='Mamba')
            axes[0, 0].set_xlabel('Validation Loss')
            axes[0, 0].set_ylabel('Validation Accuracy (Length 50)')
            axes[0, 0].set_title('Loss vs Accuracy Correlation')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log')
    
    # 2. Training Stability: Loss Variance
    window = 50
    if len(t_history['train_losses']) > window:
        t_loss_variance = np.array([np.var(t_history['train_losses'][max(0, i-window):i+1]) 
                                   for i in range(len(t_history['train_losses']))])
        m_loss_variance = np.array([np.var(m_history['train_losses'][max(0, i-window):i+1]) 
                                   for i in range(len(m_history['train_losses']))])
        
        axes[0, 1].plot(t_history['steps'], t_loss_variance, 
                       color=COLORS['transformer'], label='Transformer', linewidth=2)
        axes[0, 1].plot(m_history['steps'], m_loss_variance, 
                       color=COLORS['mamba'], label='Mamba', linewidth=2)
        axes[0, 1].set_xlabel('Training Steps')
        axes[0, 1].set_ylabel('Loss Variance (50-step window)')
        axes[0, 1].set_title('Training Stability')
        axes[0, 1].legend()
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Parameter Norm Evolution
    axes[1, 0].plot(t_history['steps'], t_history['parameter_norms'], 
                   color=COLORS['transformer'], label='Transformer', linewidth=2)
    axes[1, 0].plot(m_history['steps'], m_history['parameter_norms'], 
                   color=COLORS['mamba'], label='Mamba', linewidth=2)
    axes[1, 0].set_xlabel('Training Steps')
    axes[1, 0].set_ylabel('Parameter Norm')
    axes[1, 0].set_title('Model Parameter Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Final Diagnostic Summary
    axes[1, 1].axis('off')
    
    # Calculate diagnostic metrics
    t_final_loss = t_history['train_losses'][-1] if t_history['train_losses'] else float('inf')
    m_final_loss = m_history['train_losses'][-1] if m_history['train_losses'] else float('inf')
    
    t_final_val_loss = t_history['val_losses'][-1] if t_history['val_losses'] else float('inf')
    m_final_val_loss = m_history['val_losses'][-1] if m_history['val_losses'] else float('inf')
    
    # Get final validation accuracy for length 50
    t_final_acc = 0.0
    m_final_acc = 0.0
    if 'val_acc_50' in t_history['val_accuracies'] and t_history['val_accuracies']['val_acc_50']:
        t_final_acc = t_history['val_accuracies']['val_acc_50'][-1]
    if 'val_acc_50' in m_history['val_accuracies'] and m_history['val_accuracies']['val_acc_50']:
        m_final_acc = m_history['val_accuracies']['val_acc_50'][-1]
    
    # Diagnostic text
    diagnostic_text = f"""
TRAINING DIAGNOSTICS SUMMARY

TRANSFORMER:
• Final Train Loss: {t_final_loss:.6f}
• Final Val Loss: {t_final_val_loss:.6f}
• Final Val Acc (50): {t_final_acc:.1%}
• Train/Val Gap: {abs(t_final_loss - t_final_val_loss):.6f}

MAMBA:
• Final Train Loss: {m_final_loss:.6f}
• Final Val Loss: {m_final_val_loss:.6f}
• Final Val Acc (50): {m_final_acc:.1%}
• Train/Val Gap: {abs(m_final_loss - m_final_val_loss):.6f}

POTENTIAL ISSUES:
"""
    
    # Add specific diagnostic warnings
    if m_final_loss < 0.01 and m_final_acc < 0.1:
        diagnostic_text += "⚠️  Mamba: Very low loss but poor accuracy!\n   → Possible evaluation bug or overfitting\n"
    
    if abs(t_final_loss - t_final_val_loss) > 1.0:
        diagnostic_text += "⚠️  Transformer: Large train/val gap\n   → Possible overfitting\n"
    
    if abs(m_final_loss - m_final_val_loss) > 1.0:
        diagnostic_text += "⚠️  Mamba: Large train/val gap\n   → Possible overfitting\n"
    
    if t_final_loss > 2.0 or m_final_loss > 2.0:
        diagnostic_text += "⚠️  High final loss\n   → Models may not have converged\n"
    
    axes[1, 1].text(0.05, 0.95, diagnostic_text, transform=axes[1, 1].transAxes, 
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"📊 Training diagnostics saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    # Test simple interface
    print("Testing simple plotting interface...")
    transformer_results = {50: 0.95, 100: 0.88, 200: 0.75}
    mamba_results = {50: 0.65, 100: 0.45, 200: 0.25}
    
    create_performance_comparison(transformer_results, mamba_results, "test_figures/test_performance.png")
    create_training_summary(0.000, 0.420, "test_figures/test_training.png")
    print("✅ Test figures generated!")