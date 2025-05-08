import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap


OUTPUT_DIR = 'working/03'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def visualize_pca(pca_df, feat_types):
    """Original PCA visualization function"""
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    
    # Use a matplotlib colormap (tab10 is good for up to 10 classes)
    colors = plt.cm.tab10.colors
    
    # Plot each feature type with a distinct color
    for i, feat in enumerate(feat_types):
        subset = pca_df[pca_df['FeatType'] == feat]
        ax.scatter(subset['PC1'], subset['PC2'], 
                   label=feat, 
                   color=colors[i % len(colors)], 
                   s=50, alpha=0.7)
    
    # Label the axes and add a title and legend
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_title('PCA of Extracted Pixel Values')
    ax.legend()
    
    # Improve layout and display the plot
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/pca_visualization.png')
    #plt.show()

def visualize_color_blocks(df, lt_true_color_bands, lt_false_color_bands, at_true_color_bands, feat_types):
    """
    Create a grid of color blocks for each classification, using brightness percentiles
    from the full 9-dimensional spectral space. Uses original color values without normalization.
    """
    # Set up the figure with three columns (low tide true color, low tide false color, all tide true color)
    fig, axes = plt.subplots(len(feat_types), 3, figsize=(9, len(feat_types)),dpi=200)
    
    # If there's only one feature type, make axes 2D
    if len(feat_types) == 1:
        axes = axes.reshape(1, -1)
    
    # Combine all bands for calculation
    all_bands = lt_true_color_bands + lt_false_color_bands + at_true_color_bands
    
    # For each feature type
    for i, feat in enumerate(feat_types):
        subset = df[df['FeatType'] == feat]
        
        # Get all 6 bands together (original values)
        all_color_values = subset[all_bands].values
        
        # Calculate brightness using the original values (sum of all 6 bands)
        brightness = np.sum(all_color_values, axis=1)
        
        # Sort indices by combined brightness
        sorted_indices = np.argsort(brightness)
        
        # Number of colors in the grid
        color_rows = 8
        color_cols = 30
        n_colors = color_rows * color_cols
        
        # If we don't have enough samples
        if len(sorted_indices) < n_colors:
            # Use all available samples and repeat the last one if needed
            selected_indices = np.concatenate([
                sorted_indices,
                np.repeat(sorted_indices[-1], n_colors - len(sorted_indices))
            ])
        else:
            # Select indices based on percentiles from 1.5% to 98.5% to avoid extremes
            percentiles = np.linspace(1.5, 98.5, n_colors)
            selected_indices = np.zeros(n_colors, dtype=int)
            
            for j, percentile in enumerate(percentiles):
                # Calculate the index corresponding to this percentile
                idx = int((percentile / 100) * (len(sorted_indices) - 1))
                selected_indices[j] = sorted_indices[idx]
        
        # Extract the original true color and false color values for the selected samples
        lt_true_values = subset[lt_true_color_bands].values[selected_indices]
        lt_false_values = subset[lt_false_color_bands].values[selected_indices]
        at_true_values = subset[at_true_color_bands].values[selected_indices]
        
        # Scale for display (matplotlib requires 0-1 values for float RGB)
        # This is not normalization per se, but necessary scaling for visualization
        def scale_for_display(values):
            # Find the maximum value across all bands and samples to maintain proportion
            max_val = np.max(values)
            if max_val > 0:
                return values / max_val
            return values
        
        lt_true_display = scale_for_display(lt_true_values)
        lt_false_display = scale_for_display(lt_false_values)
        at_true_display = scale_for_display(at_true_values)
        
        # Reshape to 4x8 grid for display
        lt_true_grid = lt_true_display.reshape(color_rows, color_cols, 3)
        lt_false_grid = lt_false_display.reshape(color_rows, color_cols, 3)
        at_true_grid = at_true_display.reshape(color_rows, color_cols, 3)
        
        # Display the color grids
        axes[i, 0].imshow(lt_true_grid, interpolation='nearest')
        axes[i, 1].imshow(lt_false_grid, interpolation='nearest')
        axes[i, 2].imshow(at_true_grid, interpolation='nearest')
        
        # Remove ticks
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])
        
        # Add feature type as y-axis label
        axes[i, 0].set_ylabel(feat)
    
    # Add column titles
    axes[0, 0].set_title('Low tide True Color\n (B4=Red, B3=Green, B2=Blue)')
    axes[0, 1].set_title('Low tide False Color\n (B12=Red, B8=Green, B5=Blue)')
    axes[0, 2].set_title('All tide True Color\n (B4=Red, B3=Green, B2=Blue)')
    
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/color_blocks.png')
    #plt.show()

def main():
    # Path to the CSV with extracted pixel values
    csv_path = r'working/training-data/training-data-with-pixels.csv'
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Define the band columns to use
    # Low tide True color bands (RGB = B2, B3, B4)
    lt_true_color_bands = ['S2_LT_B2', 'S2_LT_B3', 'S2_LT_B4']
    # Low tide False color bands (RGB = B5, B8, B12)
    lt_false_color_bands = ['S2_LT_B5', 'S2_LT_B8', 'S2_LT_B12']

    # All tide True color bands (RGB = B2, B3, B4)
    at_true_color_bands = ['S2_AT_B2', 'S2_AT_B3', 'S2_AT_B4']

    # All bands
    band_cols = lt_false_color_bands + lt_false_color_bands + at_true_color_bands
    
    # Verify that the required columns exist in the DataFrame
    missing_cols = [col for col in band_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing from the CSV: {missing_cols}")
    
    # Extract the pixel values as a NumPy array
    X = df[band_cols].values
    
    # Perform PCA to reduce the 9 dimensions down to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a new DataFrame with the PCA results and feature type for plotting
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['FeatType'] = df['FeatType']
    
    # Get a list of unique feature types for color coding
    feat_types = df['FeatType'].unique()
    
    # Original PCA visualization
    print("Generating PCA visualization...")
    visualize_pca(pca_df, feat_types)
    
    # Color block visualization with 4x8 grid
    print("Generating color block visualization ...")
    visualize_color_blocks(df, lt_true_color_bands, lt_false_color_bands, at_true_color_bands, feat_types)


if __name__ == '__main__':
    main()
