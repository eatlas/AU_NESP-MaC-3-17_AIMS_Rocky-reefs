import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    # Path to the CSV with extracted pixel values
    csv_path = r'working/training-data/training-data-with-pixels.csv'
    
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_path)
    
    # Define the band columns to use (6-dimensional data)
    band_cols = ['S2_B5', 'S2_B8', 'S2_B12', 'S2_B2', 'S2_B3', 'S2_B4']
    
    # Verify that the required columns exist in the DataFrame
    missing_cols = [col for col in band_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing from the CSV: {missing_cols}")
    
    # Extract the pixel values as a NumPy array
    X = df[band_cols].values
    
    # Perform PCA to reduce the 6 dimensions down to 2
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # Create a new DataFrame with the PCA results and feature type for plotting
    pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
    pca_df['FeatType'] = df['FeatType']
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get a list of unique feature types for color coding
    feat_types = pca_df['FeatType'].unique()
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
    plt.show()

if __name__ == '__main__':
    main()
