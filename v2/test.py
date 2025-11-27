import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                            classification_report, precision_recall_curve,
                            average_precision_score)
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime
from model import AttentionZhuNet
from utils import get_data_loaders, calculate_metrics


def test_model(args):
    """
    Test the model and generate comprehensive evaluation metrics.

    Args:
        args: Command line arguments

    Returns:
        dict: Dictionary containing all test results
    """
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    print("\n" + "="*70)
    print("Loading test dataset...")
    print("="*70)

    _, _, test_loader = get_data_loaders(
        args.train_cover,  # Not used but required by function
        args.train_stego,
        args.val_cover,
        args.val_stego,
        args.test_cover,
        args.test_stego,
        args.batch_size,
        args.num_workers
    )

    # Load model
    print(f"\nLoading model from {args.model_path}...")
    model = AttentionZhuNet().to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Model loaded successfully!")

    model.eval()

    # Evaluate
    print("\n" + "="*70)
    print("Evaluating model on test set...")
    print("="*70)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0

    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for data, labels in tqdm(test_loader, desc='Testing'):
            data, labels = data.to(device), labels.to(device)

            output = model(data)
            test_loss += criterion(output, labels).item()

            # Get predictions and probabilities
            probs = torch.softmax(output, dim=1)[:, 1]  # Probability of stego class
            pred = output.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(pred.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # Calculate loss
    test_loss /= len(test_loader)

    # Calculate metrics
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)

    metrics = calculate_metrics(all_labels, all_preds, all_probs)

    print(f"\nOverall Metrics:")
    print(f"  Test Loss:  {test_loss:.6f}")
    print(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1 Score:   {metrics['f1_score']:.4f}")
    print(f"  AUC:        {metrics['auc']:.4f}")

    print(f"\nPer-Class Metrics:")
    print(f"  Cover Images:")
    print(f"    Precision: {metrics['precision_cover']:.4f}")
    print(f"    Recall:    {metrics['recall_cover']:.4f}")
    print(f"    F1 Score:  {metrics['f1_cover']:.4f}")
    print(f"\n  Stego Images:")
    print(f"    Precision: {metrics['precision_stego']:.4f}")
    print(f"    Recall:    {metrics['recall_stego']:.4f}")
    print(f"    F1 Score:  {metrics['f1_stego']:.4f}")

    # Classification report
    print("\n" + "="*70)
    print("Classification Report")
    print("="*70)
    print(classification_report(all_labels, all_preds,
                                target_names=['Cover (0)', 'Stego (1)'],
                                digits=4))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(f"{'':>12} {'Predicted Cover':>18} {'Predicted Stego':>18}")
    print(f"{'True Cover':<12} {cm[0,0]:>18} {cm[0,1]:>18}")
    print(f"{'True Stego':<12} {cm[1,0]:>18} {cm[1,1]:>18}")

    # Calculate additional metrics
    # ROC Curve
    fpr, tpr, roc_thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    # Precision-Recall Curve
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)

    # Find optimal threshold using Youden's J statistic
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = roc_thresholds[optimal_idx]

    print("\n" + "="*70)
    print(f"ROC AUC Score: {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"Average Precision Score: {avg_precision:.4f}")
    print("="*70)

    print(f"\nOptimal Threshold (Youden's J): {optimal_threshold:.4f}")
    print(f"At this threshold:")
    print(f"  True Positive Rate (Sensitivity):  {tpr[optimal_idx]:.4f}")
    print(f"  False Positive Rate:                {fpr[optimal_idx]:.4f}")
    print(f"  True Negative Rate (Specificity):  {1-fpr[optimal_idx]:.4f}")

    # Calculate metrics at optimal threshold
    preds_optimal = (all_probs >= optimal_threshold).astype(int)
    metrics_optimal = calculate_metrics(all_labels, preds_optimal, all_probs)
    print(f"  Accuracy at optimal threshold:      {metrics_optimal['accuracy']:.4f}")

    # Create visualizations
    create_test_visualizations(
        all_labels, all_preds, all_probs, cm,
        fpr, tpr, roc_auc, optimal_threshold, optimal_idx,
        precision_curve, recall_curve, avg_precision,
        args.output_dir
    )

    # Save results
    save_test_results(
        args, test_loss, metrics, cm, roc_auc, avg_precision,
        optimal_threshold, metrics_optimal, all_labels, all_preds, all_probs
    )

    return {
        'test_loss': test_loss,
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'optimal_threshold': optimal_threshold,
        'metrics_at_optimal': metrics_optimal
    }


def create_test_visualizations(all_labels, all_preds, all_probs, cm,
                               fpr, tpr, roc_auc, optimal_threshold, optimal_idx,
                               precision_curve, recall_curve, avg_precision,
                               output_dir):
    """Create comprehensive test visualizations."""

    # Create figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))

    # Subplot 1: ROC Curve
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random Classifier (AUC = 0.5)')
    ax1.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', s=100,
                zorder=5, label=f'Optimal Threshold = {optimal_threshold:.4f}')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
    ax1.legend(loc="lower right")
    ax1.grid(alpha=0.3)

    # Subplot 2: Precision-Recall Curve
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(recall_curve, precision_curve, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.4f})')
    ax2.axhline(y=0.5, color='navy', linestyle='--', lw=2,
                label='Baseline (AP = 0.5)')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax2.legend(loc="lower left")
    ax2.grid(alpha=0.3)

    # Subplot 3: Confusion Matrix
    ax3 = plt.subplot(2, 3, 3)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                xticklabels=['Cover', 'Stego'],
                yticklabels=['Cover', 'Stego'],
                cbar_kws={'label': 'Count'})
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

    # Add percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            ax3.text(j+0.5, i+0.7, f'({cm[i,j]/total*100:.1f}%)',
                    ha='center', va='center', color='gray', fontsize=9)

    # Subplot 4: Probability Distribution
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(all_probs[all_labels == 0], bins=50, alpha=0.6,
             label='Cover (True Negative)', color='blue', edgecolor='black')
    ax4.hist(all_probs[all_labels == 1], bins=50, alpha=0.6,
             label='Stego (True Positive)', color='red', edgecolor='black')
    ax4.axvline(x=0.5, color='green', linestyle='--', linewidth=2,
                label='Default Threshold (0.5)')
    ax4.axvline(x=optimal_threshold, color='orange', linestyle='--', linewidth=2,
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    ax4.set_xlabel('Predicted Probability', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3, axis='y')

    # Subplot 5: Normalized Confusion Matrix
    ax5 = plt.subplot(2, 3, 5)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax5,
                xticklabels=['Cover', 'Stego'],
                yticklabels=['Cover', 'Stego'],
                cbar_kws={'label': 'Percentage'})
    ax5.set_ylabel('True Label', fontsize=12)
    ax5.set_xlabel('Predicted Label', fontsize=12)
    ax5.set_title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')

    # Subplot 6: Error Analysis
    ax6 = plt.subplot(2, 3, 6)

    # Calculate error rates
    cover_as_stego = cm[0, 1]  # False positives
    stego_as_cover = cm[1, 0]  # False negatives
    correct_cover = cm[0, 0]
    correct_stego = cm[1, 1]

    categories = ['Correct\nCover', 'Cover as\nStego\n(FP)', 'Correct\nStego', 'Stego as\nCover\n(FN)']
    values = [correct_cover, cover_as_stego, correct_stego, stego_as_cover]
    colors = ['green', 'red', 'green', 'red']

    bars = ax6.bar(categories, values, color=colors, alpha=0.6, edgecolor='black')
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_title('Error Analysis', fontsize=14, fontweight='bold')
    ax6.grid(alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}\n({value/total*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    # Save plot
    output_path = os.path.join(output_dir, 'test_results.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plots saved to '{output_path}'")

    if args.show_plot:
        plt.show()

    plt.close()


def save_test_results(args, test_loss, metrics, cm, roc_auc, avg_precision,
                     optimal_threshold, metrics_optimal, all_labels, all_preds, all_probs):
    """Save test results to files."""

    os.makedirs(args.output_dir, exist_ok=True)

    # Save as JSON
    results_dict = {
        'model_path': args.model_path,
        'test_cover_path': args.test_cover,
        'test_stego_path': args.test_stego,
        'timestamp': datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'metrics': {k: float(v) for k, v in metrics.items()},
        'confusion_matrix': cm.tolist(),
        'roc_auc': float(roc_auc),
        'average_precision': float(avg_precision),
        'optimal_threshold': float(optimal_threshold),
        'metrics_at_optimal_threshold': {k: float(v) for k, v in metrics_optimal.items()},
    }

    json_path = os.path.join(args.output_dir, 'test_results.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=4)

    # Save as text file
    txt_path = os.path.join(args.output_dir, 'test_results.txt')
    with open(txt_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("STEGANALYSIS TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Cover path: {args.test_cover}\n")
        f.write(f"Stego path: {args.test_stego}\n\n")

        f.write(f"Test Loss: {test_loss:.6f}\n\n")

        f.write("Overall Metrics:\n")
        f.write(f"  Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"  Precision:  {metrics['precision']:.4f}\n")
        f.write(f"  Recall:     {metrics['recall']:.4f}\n")
        f.write(f"  F1 Score:   {metrics['f1_score']:.4f}\n")
        f.write(f"  AUC:        {metrics['auc']:.4f}\n")
        f.write(f"  Avg Precision: {avg_precision:.4f}\n\n")

        f.write("Per-Class Metrics:\n")
        f.write(f"  Cover Images:\n")
        f.write(f"    Precision: {metrics['precision_cover']:.4f}\n")
        f.write(f"    Recall:    {metrics['recall_cover']:.4f}\n")
        f.write(f"    F1 Score:  {metrics['f1_cover']:.4f}\n\n")
        f.write(f"  Stego Images:\n")
        f.write(f"    Precision: {metrics['precision_stego']:.4f}\n")
        f.write(f"    Recall:    {metrics['recall_stego']:.4f}\n")
        f.write(f"    F1 Score:  {metrics['f1_stego']:.4f}\n\n")

        f.write(f"Optimal Threshold: {optimal_threshold:.4f}\n")
        f.write(f"Accuracy at optimal threshold: {metrics_optimal['accuracy']:.4f}\n\n")

        f.write("Confusion Matrix:\n")
        f.write(f"{'':>12} {'Predicted Cover':>18} {'Predicted Stego':>18}\n")
        f.write(f"{'True Cover':<12} {cm[0,0]:>18} {cm[0,1]:>18}\n")
        f.write(f"{'True Stego':<12} {cm[1,0]:>18} {cm[1,1]:>18}\n")
        f.write("\n" + "="*70 + "\n")

    print(f"✓ Results saved to '{txt_path}' and '{json_path}'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Steganalysis Classifier (AttentionZhuNet)')

    # Machine type
    parser.add_argument(
        "--machine",
        type=str,
        default="server",
        choices=["local", "server"],
        help="Machine type: 'local' or 'server' (default: server)"
    )

    args, remaining = parser.parse_known_args()

    # Set paths based on machine type
    if args.machine == "local":
        default_train_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/train'
        default_train_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/train'
        default_val_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/val'
        default_val_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/val'
        default_test_cover = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/test'
        default_test_stego = r'/Users/dmitryhoma/Projects/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/test'
    else:  # server
        default_train_cover = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/train'
        default_train_stego = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/train'
        default_val_cover = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/val'
        default_val_stego = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/val'
        default_test_cover = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/cover/test'
        default_test_stego = r'/root/datasets/ready_to_use/GBRASNET/BOSSbase-1.01/stego/S-UNIWARD/0.4bpp/test'

    # Data paths
    parser.add_argument("--train_cover", default=default_train_cover, help="Path to training cover images")
    parser.add_argument("--train_stego", default=default_train_stego, help="Path to training stego images")
    parser.add_argument("--val_cover", default=default_val_cover, help="Path to validation cover images")
    parser.add_argument("--val_stego", default=default_val_stego, help="Path to validation stego images")
    parser.add_argument("--test_cover", default=default_test_cover, help="Path to test cover images")
    parser.add_argument("--test_stego", default=default_test_stego, help="Path to test stego images")

    # Model and testing parameters
    parser.add_argument("--model_path", type=str, default="./checkpoints/final_model.pth",
                       help="Path to trained model checkpoint")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--output_dir", type=str, default="./test_results",
                       help="Output directory for results")
    parser.add_argument("--show_plot", action="store_true",
                       help="Show plot after testing")

    args = parser.parse_args()

    results = test_model(args)
