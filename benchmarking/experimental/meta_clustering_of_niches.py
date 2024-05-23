#!/usr/bin/env python

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def load_centroids(spot_directory):
    centroids = []
    for file in os.listdir(spot_directory):
        if file.startswith('centroid_') and file.endswith('.csv'):
            try:
                df = pd.read_csv(os.path.join(spot_directory, file))
                if not df.empty:
                    centroids.append(df)
            except pd.errors.EmptyDataError:
                print(f"Warning: {file} is empty and will be skipped.")
    return centroids

def combine_centroids(centroids):
    if centroids:
        try:
            combined = pd.concat(centroids, ignore_index=True)
            return combined
        except ValueError as e:
            print(f"Error combining data: {e}")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def perform_meta_clustering(data, n_clusters=10):
    if not data.empty:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        meta_labels = kmeans.fit_predict(data)
        return meta_labels, kmeans.cluster_centers_
    else:
        return [], np.array([])

def main():
    if len(sys.argv) > 1:
        spot_directory = sys.argv[1]
    else:
        print("Please provide the directory path as a command-line argument.")
        sys.exit(1)

    centroids = load_centroids(spot_directory)
    combined_centroids = combine_centroids(centroids)

    if not combined_centroids.empty:
        meta_labels, cluster_centers = perform_meta_clustering(combined_centroids)
        combined_centroids['meta_cluster'] = meta_labels
        combined_centroids.to_csv('meta_clustering_results.csv', index=False)

        # Perform PCA
        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(combined_centroids.drop('meta_cluster', axis=1))
        plt.figure(figsize=(10, 8))
        plt.scatter(pca_results[:, 0], pca_results[:, 1], c=meta_labels, cmap='Spectral', s=50)
        plt.colorbar()
        plt.title('PCA Projection of Meta-Clusters')
        plt.savefig(os.path.join(spot_directory, 'pca_projection.png'))
        plt.show()

        # Perform UMAP
        reducer = umap.UMAP(random_state=42)
        umap_results = reducer.fit_transform(combined_centroids.drop('meta_cluster', axis=1))
        plt.figure(figsize=(10, 8))
        plt.scatter(umap_results[:, 0], umap_results[:, 1], c=meta_labels, cmap='Spectral', s=50)
        plt.colorbar()
        plt.title('UMAP Projection of Meta-Clusters')
        plt.savefig(os.path.join(spot_directory, 'umap_projection.png'))
        plt.show()

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_results = tsne.fit_transform(combined_centroids.drop('meta_cluster', axis=1))
        plt.figure(figsize=(10, 8))
        plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=meta_labels, cmap='Spectral', s=50)
        plt.colorbar()
        plt.title('t-SNE Projection of Meta-Clusters')
        plt.savefig(os.path.join(spot_directory, 'tsne_projection.png'))
        plt.show()

    else:
        print("No data available for meta-clustering.")

if __name__ == '__main__':
    main()
