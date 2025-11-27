import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd


def load_metrics(metrics_dir):
    """
    Load metrics from JSON file.

    Args:
        metrics_dir: Directory containing metrics.json

    Returns:
        dict: Dictionary containing train, valid, and best metrics
    """
    json_path = os.path.join(metrics_dir, 'metrics.json')

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Metrics file not found: {json_path}")

    with open(json_path, 'r') as f:
        metrics = json.load(f)

    return metrics


def load_metrics_from_csv(metrics_dir):
    """
    Load metrics from CSV files (alternative to JSON).

    Args:
        metrics_dir: Directory containing CSV files

    Returns:
        tuple: (train_df, valid_df) DataFrames
    """
    train_csv = os.path.join(metrics_dir, 'train_metrics.csv')
    valid_csv = os.path.join(metrics_dir, 'valid_metrics.csv')

    train_df = None
    valid_df = None

    if os.path.exists(train_csv):
        train_df = pd.read_csv(train_csv)

    if os.path.exists(valid_csv):
        valid_df = pd.read_csv(valid_csv)

    return train_df, valid_df


def plot_loss_curves(metrics, save_path=None):
    """
    Plot training and validation loss curves.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save the plot (optional)
    """
    train = metrics['train']
    valid = metrics['valid']

    plt.figure(figsize=(12, 6))

    # Plot losses
    plt.plot(train['epoch'], train['loss'], 'b-', linewidth=2, label='Training Loss', marker='o', markersize=4)

    if valid['epoch']:
        plt.plot(valid['epoch'], valid['loss'], 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=4)

        # Mark best validation loss
        best_loss = metrics['best']['best_val_loss']
        best_epoch = metrics['best']['best_val_loss_epoch']
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                   label=f'Best Val Loss @ Epoch {best_epoch}')
        plt.scatter([best_epoch], [best_loss], color='green', s=100, zorder=5, marker='*')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Training and Validation Loss', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Loss curves saved to {save_path}")

    return plt.gcf()


def plot_accuracy_curves(metrics, save_path=None):
    """
    Plot training and validation accuracy curves.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save the plot (optional)
    """
    train = metrics['train']
    valid = metrics['valid']

    plt.figure(figsize=(12, 6))

    # Plot accuracies
    plt.plot(train['epoch'], [acc * 100 for acc in train['accuracy']],
             'b-', linewidth=2, label='Training Accuracy', marker='o', markersize=4)

    if valid['epoch']:
        plt.plot(valid['epoch'], [acc * 100 for acc in valid['accuracy']],
                'r-', linewidth=2, label='Validation Accuracy', marker='s', markersize=4)

        # Mark best validation accuracy
        best_acc = metrics['best']['best_val_acc']
        best_epoch = metrics['best']['best_val_acc_epoch']
        plt.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5,
                   label=f'Best Val Acc @ Epoch {best_epoch}')
        plt.scatter([best_epoch], [best_acc * 100], color='green', s=100, zorder=5, marker='*')

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Accuracy curves saved to {save_path}")

    return plt.gcf()


def plot_all_metrics(metrics, save_path=None):
    """
    Plot all metrics (Precision, Recall, F1) for validation set.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save the plot (optional)
    """
    valid = metrics['valid']

    if not valid['epoch']:
        print("No validation metrics to plot")
        return None

    plt.figure(figsize=(14, 6))

    # Plot all metrics
    plt.plot(valid['epoch'], [p * 100 for p in valid['precision']],
             'b-', linewidth=2, label='Precision', marker='o', markersize=4)
    plt.plot(valid['epoch'], [r * 100 for r in valid['recall']],
             'g-', linewidth=2, label='Recall', marker='s', markersize=4)
    plt.plot(valid['epoch'], [f * 100 for f in valid['f1_score']],
             'r-', linewidth=2, label='F1 Score', marker='^', markersize=4)
    plt.plot(valid['epoch'], [a * 100 for a in valid['accuracy']],
             'm-', linewidth=2, label='Accuracy', marker='d', markersize=4)

    if 'auc' in valid and valid['auc']:
        plt.plot(valid['epoch'], [auc * 100 for auc in valid['auc']],
                'c-', linewidth=2, label='AUC', marker='*', markersize=6)

    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Score (%)', fontsize=14)
    plt.title('Validation Metrics Over Time', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(alpha=0.3)
    plt.ylim([0, 105])
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ All metrics plot saved to {save_path}")

    return plt.gcf()


def plot_per_class_metrics(metrics, save_path=None):
    """
    Plot per-class metrics (Cover vs Stego) for validation set.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save the plot (optional)
    """
    valid = metrics['valid']

    if not valid['epoch']:
        print("No validation metrics to plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Cover metrics
    ax1 = axes[0]
    ax1.plot(valid['epoch'], [p * 100 for p in valid['precision_cover']],
             'b-', linewidth=2, label='Precision', marker='o', markersize=4)
    ax1.plot(valid['epoch'], [r * 100 for r in valid['recall_cover']],
             'g-', linewidth=2, label='Recall', marker='s', markersize=4)
    ax1.plot(valid['epoch'], [f * 100 for f in valid['f1_cover']],
             'r-', linewidth=2, label='F1 Score', marker='^', markersize=4)
    ax1.set_xlabel('Epoch', fontsize=14)
    ax1.set_ylabel('Score (%)', fontsize=14)
    ax1.set_title('Cover Image Metrics', fontsize=16, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 105])

    # Stego metrics
    ax2 = axes[1]
    ax2.plot(valid['epoch'], [p * 100 for p in valid['precision_stego']],
             'b-', linewidth=2, label='Precision', marker='o', markersize=4)
    ax2.plot(valid['epoch'], [r * 100 for r in valid['recall_stego']],
             'g-', linewidth=2, label='Recall', marker='s', markersize=4)
    ax2.plot(valid['epoch'], [f * 100 for f in valid['f1_stego']],
             'r-', linewidth=2, label='F1 Score', marker='^', markersize=4)
    ax2.set_xlabel('Epoch', fontsize=14)
    ax2.set_ylabel('Score (%)', fontsize=14)
    ax2.set_title('Stego Image Metrics', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Per-class metrics plot saved to {save_path}")

    return fig


def plot_learning_rate(metrics, save_path=None):
    """
    Plot learning rate over epochs.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save the plot (optional)
    """
    train = metrics['train']

    if 'learning_rate' not in train or not train['learning_rate']:
        print("No learning rate data to plot")
        return None

    plt.figure(figsize=(12, 6))

    plt.plot(train['epoch'], train['learning_rate'], 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Learning Rate', fontsize=14)
    plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
    plt.yscale('log')
    plt.grid(alpha=0.3, which='both')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Learning rate plot saved to {save_path}")

    return plt.gcf()


def plot_comprehensive_dashboard(metrics, save_path=None):
    """
    Create a comprehensive dashboard with all metrics in one figure.

    Args:
        metrics: Dictionary containing metrics
        save_path: Path to save the plot (optional)
    """
    train = metrics['train']
    valid = metrics['valid']

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(train['epoch'], train['loss'], 'b-', linewidth=2, label='Train', marker='o', markersize=3)
    if valid['epoch']:
        ax1.plot(valid['epoch'], valid['loss'], 'r-', linewidth=2, label='Valid', marker='s', markersize=3)
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss', fontsize=10)
    ax1.set_title('Loss Curves', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 2. Accuracy curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(train['epoch'], [a * 100 for a in train['accuracy']],
             'b-', linewidth=2, label='Train', marker='o', markersize=3)
    if valid['epoch']:
        ax2.plot(valid['epoch'], [a * 100 for a in valid['accuracy']],
                'r-', linewidth=2, label='Valid', marker='s', markersize=3)
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Accuracy (%)', fontsize=10)
    ax2.set_title('Accuracy Curves', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])

    # 3. Learning Rate
    ax3 = fig.add_subplot(gs[0, 2])
    if 'learning_rate' in train and train['learning_rate']:
        ax3.plot(train['epoch'], train['learning_rate'], 'g-', linewidth=2, marker='o', markersize=3)
        ax3.set_yscale('log')
    ax3.set_xlabel('Epoch', fontsize=10)
    ax3.set_ylabel('Learning Rate', fontsize=10)
    ax3.set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, which='both')

    if valid['epoch']:
        # 4. All validation metrics
        ax4 = fig.add_subplot(gs[1, :])
        ax4.plot(valid['epoch'], [p * 100 for p in valid['precision']],
                'b-', linewidth=2, label='Precision', marker='o', markersize=3)
        ax4.plot(valid['epoch'], [r * 100 for r in valid['recall']],
                'g-', linewidth=2, label='Recall', marker='s', markersize=3)
        ax4.plot(valid['epoch'], [f * 100 for f in valid['f1_score']],
                'r-', linewidth=2, label='F1 Score', marker='^', markersize=3)
        ax4.plot(valid['epoch'], [a * 100 for a in valid['accuracy']],
                'm-', linewidth=2, label='Accuracy', marker='d', markersize=3)
        if 'auc' in valid and valid['auc']:
            ax4.plot(valid['epoch'], [auc * 100 for auc in valid['auc']],
                    'c-', linewidth=2, label='AUC', marker='*', markersize=5)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('Score (%)', fontsize=10)
        ax4.set_title('Validation Metrics', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=9, loc='lower right')
        ax4.grid(alpha=0.3)
        ax4.set_ylim([0, 105])

        # 5. Cover class metrics
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(valid['epoch'], [p * 100 for p in valid['precision_cover']],
                'b-', linewidth=2, label='Precision', marker='o', markersize=3)
        ax5.plot(valid['epoch'], [r * 100 for r in valid['recall_cover']],
                'g-', linewidth=2, label='Recall', marker='s', markersize=3)
        ax5.plot(valid['epoch'], [f * 100 for f in valid['f1_cover']],
                'r-', linewidth=2, label='F1 Score', marker='^', markersize=3)
        ax5.set_xlabel('Epoch', fontsize=10)
        ax5.set_ylabel('Score (%)', fontsize=10)
        ax5.set_title('Cover Image Metrics', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)
        ax5.set_ylim([0, 105])

        # 6. Stego class metrics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(valid['epoch'], [p * 100 for p in valid['precision_stego']],
                'b-', linewidth=2, label='Precision', marker='o', markersize=3)
        ax6.plot(valid['epoch'], [r * 100 for r in valid['recall_stego']],
                'g-', linewidth=2, label='Recall', marker='s', markersize=3)
        ax6.plot(valid['epoch'], [f * 100 for f in valid['f1_stego']],
                'r-', linewidth=2, label='F1 Score', marker='^', markersize=3)
        ax6.set_xlabel('Epoch', fontsize=10)
        ax6.set_ylabel('Score (%)', fontsize=10)
        ax6.set_title('Stego Image Metrics', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(alpha=0.3)
        ax6.set_ylim([0, 105])

        # 7. Best metrics summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        best = metrics['best']
        summary_text = "BEST VALIDATION METRICS\n" + "="*35 + "\n\n"
        summary_text += f"Best Accuracy:\n"
        summary_text += f"  {best['best_val_acc']*100:.2f}% @ Epoch {best['best_val_acc_epoch']}\n\n"
        summary_text += f"Best Loss:\n"
        summary_text += f"  {best['best_val_loss']:.4f} @ Epoch {best['best_val_loss_epoch']}\n\n"
        summary_text += f"Best F1 Score:\n"
        summary_text += f"  {best['best_val_f1']*100:.2f}% @ Epoch {best['best_val_f1_epoch']}\n\n"

        if 'training_duration' in metrics:
            summary_text += f"Training Duration:\n  {metrics['training_duration']}"

        ax7.text(0.1, 0.95, summary_text, transform=ax7.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    fig.suptitle('Training Metrics Dashboard', fontsize=18, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comprehensive dashboard saved to {save_path}")

    return fig


def compare_metrics_across_runs(metrics_dirs, labels=None, metric_name='accuracy', save_path=None):
    """
    Compare a specific metric across multiple training runs.

    Args:
        metrics_dirs: List of directories containing metrics.json files
        labels: List of labels for each run (optional)
        metric_name: Name of the metric to compare
        save_path: Path to save the plot (optional)
    """
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(metrics_dirs))]

    plt.figure(figsize=(14, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_dirs)))

    for i, (metrics_dir, label) in enumerate(zip(metrics_dirs, labels)):
        try:
            metrics = load_metrics(metrics_dir)
            valid = metrics['valid']

            if valid['epoch'] and metric_name in valid:
                values = [v * 100 if metric_name != 'loss' else v for v in valid[metric_name]]
                plt.plot(valid['epoch'], values, linewidth=2, label=label,
                        marker='o', markersize=3, color=colors[i])
        except Exception as e:
            print(f"Error loading metrics from {metrics_dir}: {e}")

    plt.xlabel('Epoch', fontsize=14)
    ylabel = f'{metric_name.replace("_", " ").title()}'
    if metric_name != 'loss':
        ylabel += ' (%)'
    plt.ylabel(ylabel, fontsize=14)
    plt.title(f'Comparison: {metric_name.replace("_", " ").title()}', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison plot saved to {save_path}")

    return plt.gcf()


def plot_metrics_from_csv(train_csv, valid_csv, output_dir):
    """
    Plot metrics directly from CSV files.

    Args:
        train_csv: Path to training metrics CSV
        valid_csv: Path to validation metrics CSV
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_csv) if os.path.exists(train_csv) else None
    valid_df = pd.read_csv(valid_csv) if os.path.exists(valid_csv) else None

    # Plot loss
    plt.figure(figsize=(12, 6))
    if train_df is not None:
        plt.plot(train_df['epoch'], train_df['loss'], 'b-', linewidth=2,
                label='Training Loss', marker='o', markersize=4)
    if valid_df is not None:
        plt.plot(valid_df['epoch'], valid_df['loss'], 'r-', linewidth=2,
                label='Validation Loss', marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.title('Loss Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_from_csv.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(12, 6))
    if train_df is not None:
        plt.plot(train_df['epoch'], train_df['accuracy'] * 100, 'b-', linewidth=2,
                label='Training Accuracy', marker='o', markersize=4)
    if valid_df is not None:
        plt.plot(valid_df['epoch'], valid_df['accuracy'] * 100, 'r-', linewidth=2,
                label='Validation Accuracy', marker='s', markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.title('Accuracy Curves', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_from_csv.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Plots from CSV saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Plot Training Metrics')

    parser.add_argument('--metrics_dir', type=str, default='./checkpoints',
                       help='Directory containing metrics.json file')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    parser.add_argument('--plot_type', type=str, default='all',
                       choices=['loss', 'accuracy', 'metrics', 'per_class',
                               'lr', 'dashboard', 'all'],
                       help='Type of plot to generate')
    parser.add_argument('--compare_dirs', type=str, nargs='+', default=None,
                       help='Multiple directories to compare metrics across runs')
    parser.add_argument('--compare_labels', type=str, nargs='+', default=None,
                       help='Labels for comparison runs')
    parser.add_argument('--compare_metric', type=str, default='accuracy',
                       help='Metric to compare across runs')
    parser.add_argument('--show', action='store_true',
                       help='Show plots instead of just saving them')
    parser.add_argument('--use_csv', action='store_true',
                       help='Load metrics from CSV files instead of JSON')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Handle comparison mode
    if args.compare_dirs:
        print("\n" + "="*70)
        print("Comparing metrics across multiple runs...")
        print("="*70)

        compare_metrics_across_runs(
            args.compare_dirs,
            args.compare_labels,
            args.compare_metric,
            os.path.join(args.output_dir, f'comparison_{args.compare_metric}.png')
        )

        if args.show:
            plt.show()

        return

    # Load metrics
    print("\n" + "="*70)
    print(f"Loading metrics from {args.metrics_dir}...")
    print("="*70)

    try:
        if args.use_csv:
            train_csv = os.path.join(args.metrics_dir, 'train_metrics.csv')
            valid_csv = os.path.join(args.metrics_dir, 'valid_metrics.csv')
            plot_metrics_from_csv(train_csv, valid_csv, args.output_dir)
            return

        metrics = load_metrics(args.metrics_dir)
        print(f"✓ Metrics loaded successfully")

        # Generate plots based on type
        print(f"\nGenerating {args.plot_type} plot(s)...")

        if args.plot_type in ['loss', 'all']:
            plot_loss_curves(metrics,
                           os.path.join(args.output_dir, 'loss_curves.png'))

        if args.plot_type in ['accuracy', 'all']:
            plot_accuracy_curves(metrics,
                               os.path.join(args.output_dir, 'accuracy_curves.png'))

        if args.plot_type in ['metrics', 'all']:
            plot_all_metrics(metrics,
                           os.path.join(args.output_dir, 'all_metrics.png'))

        if args.plot_type in ['per_class', 'all']:
            plot_per_class_metrics(metrics,
                                 os.path.join(args.output_dir, 'per_class_metrics.png'))

        if args.plot_type in ['lr', 'all']:
            plot_learning_rate(metrics,
                             os.path.join(args.output_dir, 'learning_rate.png'))

        if args.plot_type in ['dashboard', 'all']:
            plot_comprehensive_dashboard(metrics,
                                       os.path.join(args.output_dir, 'dashboard.png'))

        print(f"\n✓ All plots saved to {args.output_dir}")

        # Display best metrics
        print("\n" + "="*70)
        print("BEST METRICS")
        print("="*70)
        best = metrics['best']
        print(f"Best Validation Accuracy:  {best['best_val_acc']*100:.2f}% @ Epoch {best['best_val_acc_epoch']}")
        print(f"Best Validation Loss:      {best['best_val_loss']:.4f} @ Epoch {best['best_val_loss_epoch']}")
        print(f"Best Validation F1 Score:  {best['best_val_f1']*100:.2f}% @ Epoch {best['best_val_f1_epoch']}")

        if 'training_duration' in metrics:
            print(f"\nTraining Duration: {metrics['training_duration']}")
        print("="*70)

        if args.show:
            plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
