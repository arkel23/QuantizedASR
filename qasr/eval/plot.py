import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import numpy as np


def plot_tone_confusion_matrix(cf_matrix, fp, save_to_wandb=False, dpi=160):
    """
    Plot tone confusion matrix
    
    Args:
        cf_matrix: NxN numpy array from compute_ter()
        tone_labels: List of tone labels ['1', '2', '3', '4', '5']
        fp: Optional path to save figure
    """
    tones = cf_matrix.shape[0]
    tone_labels = [str(i + 1) for i in range(tones)]

    fig = plt.figure(figsize=(8, 6))

    percent_matrix = np.zeros_like(cf_matrix, dtype=np.float32)
    for i, row in enumerate(cf_matrix):
        row_norm = 100 * row / sum(row)
        percent_matrix[i] = row_norm

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2f}%".format(value) for value in percent_matrix.flatten()]
    labels = [f"{c}\n{p}" for c, p in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cf_matrix.shape)

    # Use seaborn heatmap
    sns.heatmap(percent_matrix, 
                annot=labels,
                # annot=True,  # Show numbers
                fmt='',
                # fmt='d',  # Integer format
                # fmt='.2%', # percent
                # cmap='YlOrRd',
                # cmap='YlOrBr',
                # cmap='coolwarm',
                cmap='Blues',
                xticklabels=tone_labels,
                yticklabels=tone_labels,
                cbar_kws={'label': 'Percentage (%)'})
    
    plt.xlabel('Predicted Tone')
    plt.ylabel('Reference Tone')
    plt.title('Tone Confusion Matrix')
    plt.tight_layout()

    plt.savefig(fp, dpi=dpi, bbox_inches='tight')

    if save_to_wandb:
        wandb.log({'tone_confusion_matrix': wandb.Image(fig)})

    return 0
