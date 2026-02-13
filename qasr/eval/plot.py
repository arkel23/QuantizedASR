import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def plot_tone_confusion_matrix(confusion_matrix, fp, save_to_wandb=False, dpi=160):
    """
    Plot tone confusion matrix
    
    Args:
        confusion_matrix: NxN numpy array from compute_ter()
        tone_labels: List of tone labels ['1', '2', '3', '4', '5']
        fp: Optional path to save figure
    """
    tones = confusion_matrix.shape[0]
    tone_labels = [str(i + 1) for i in range(tones)]

    fig = plt.figure(figsize=(8, 6))
    
    # Use seaborn heatmap
    sns.heatmap(confusion_matrix, 
                annot=True,  # Show numbers
                fmt='d',  # Integer format
                cmap='YlOrRd',
                xticklabels=tone_labels,
                yticklabels=tone_labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Tone')
    plt.ylabel('Reference Tone')
    plt.title('Tone Confusion Matrix')
    plt.tight_layout()

    plt.savefig(fp, dpi=dpi, bbox_inches='tight')

    if save_to_wandb:
        wandb.log({'tone_confusion_matrix': wandb.Image(fig)})

    return 0
