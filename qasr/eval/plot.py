'''
# import matplotlib.pyplot as plt
# import seaborn as sns

def plot_tone_confusion_matrix(confusion_matrix, save_path=None):
    """
    Plot tone confusion matrix
    
    Args:
        confusion_matrix: NxN numpy array from compute_ter()
        tone_labels: List of tone labels ['1', '2', '3', '4', '5']
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(8, 6))
    
    # Use seaborn heatmap
    sns.heatmap(confusion_matrix, 
                annot=True,  # Show numbers
                fmt='d',  # Integer format
                cmap='YlOrRd',
                # xticklabels=tone_labels,
                # yticklabels=tone_labels,
                cbar_kws={'label': 'Count'})
    
    plt.xlabel('Predicted Tone')
    plt.ylabel('Reference Tone')
    plt.title('Tone Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
'''
