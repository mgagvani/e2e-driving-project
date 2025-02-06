import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix():
    # Confusion matrix data
    confusion_matrix = np.array([
        [0.3206712604, 0.02224458195, 0.2895650864],
        [0.3378834724, 0.02619451098, 0.2589342594],
        [0.3172439635, 0.0276527144, 0.3514472544]
    ])

    # Box array data for the "All" row
    box_array = np.array([0.0416210182, 0.005532090086, 0.09645161033])

    all_data = np.concatenate([confusion_matrix.flatten(), box_array])
    vmin = all_data.min()
    vmax = all_data.max()

    # Labels for the confusion matrix
    categories = ["Left", "Straight", "Right"]

    # Combined visualization
    plt.figure(figsize=(11, 17))  # Increased height for better aspect ratio
    sns.set_context("poster")

    cmap = 'plasma_r'  

    # Plot the confusion matrix
    plt.subplot(2, 1, 1)
    heatmap1 = sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt=".3f",
        cmap=cmap,  # Changed colormap to yellows and oranges
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={'label': '$L_{waypoint}$ - MSE'},
        square=True,  # Ensure squares for consistent box sizes
        vmin=vmin,          # Set minimum for color scale
        vmax=vmax           # Set maximum for color scale
    )
    plt.title("Driving Scenario vs. Input Camera")
    plt.xlabel("Turn Type")
    plt.ylabel("Camera Used")

    # Plot the box array
    plt.subplot(2, 1, 2)
    heatmap2 = sns.heatmap(
        box_array.reshape(1, -1),
        annot=True,
        fmt=".3f",
        cmap=cmap,  # Changed colormap to match heatmap1
        xticklabels=categories,
        yticklabels=["All"],
        cbar_kws={'label': '$L_{waypoint}$ - MSE'},
        square=True,  # Make boxes square
        linewidths=0.5,  # Add lines for better separation
        linecolor='white',
        vmin=vmin,          # Use the same vmin as heatmap1
        vmax=vmax           # Use the same vmax as heatmap1
    )
    plt.title("Multi-Camera")
    plt.xlabel("Turn Type")
    plt.ylabel("All Cameras")

    plt.tight_layout()
    sns.set_context("poster")
    plt.savefig("results/confusion_matrix.png")

if __name__ == "__main__":
    confusion_matrix()