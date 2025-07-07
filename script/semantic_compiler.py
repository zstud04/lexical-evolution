"""
Class for managing matrix operations for a single embeddings matrix instance
"""
import argparse
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.optimize import curve_fit
from sklearn.decomposition import SparsePCA


WORD_BASE_DIR = './staging'

class EmbeddingsMatrix:
    def __init__(self, matrix_dir:str, word:str):

        self.word = word
      
        self.matrix_dir = matrix_dir
        self.decade = self.__extract_decade()

        self.matrix = self.load_embeddings_matrix()
        self.senses = self.__load_metadata()[1:]


        ##get singular values/matrices, components
        self.U, self.S, self.VT = np.linalg.svd(self.matrix)
        self.pca = self.__pca_compute()

        ####Metrics properties#####
        self.spectral_diversity = self.calculate_spectral_diversity()
        self.n_senses = self.count_senses()
        self.entropy = self.calculate_eigenvalue_entropy()
        self.condition_number = self.calculate_condition_number()
        #self.variance_contribution_slope = self.variance_contribution_slope()



    def load_embeddings_matrix(self):
        # Construct the file path to the numpy matrix
        file_path = os.path.join(self.matrix_dir, f"{self.word}_embeddings.npy")
        # Load and return the matrix if it exists
        if os.path.exists(file_path):
            return np.load(file_path)
        else:
            raise FileNotFoundError(f"No embeddings matrix found for the word '{self.word}' at '{file_path}'")
        

    def __extract_decade(self):
        """
        Regex method to extract decade from path str
        """
        match = re.search(r'[^/]+$', self.matrix_dir)
        if match:
            return int(match.group(0))
        return None


    def __load_metadata(self):
        """
        Load word senses for a given word instance
        """
        file_path = os.path.join(self.matrix_dir, f"{self.word}_metadata.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, header=None)  # Assuming no header in the file
            # Split each row on spaces and extract the second word (index 1)
            senses = df[0].apply(lambda x: x.split()[1] if len(x.split()) > 1 else None).dropna().unique().tolist()
            return senses
        else:
            raise FileNotFoundError(f"No metadata file found for the word '{self.word}' at '{file_path}'")



    def plot_pc_directions(self, top_n=5):
        """
        Plot the directioons of the top n principal components for the embeddings matrix, with associated senses
        Args:
            top_n (int): number of components to plot
        """
        # Select the top_n principal components
        components = self.VT[:top_n, :]

        # Plot a circle
        circle = plt.Circle((0, 0), 1, color='gray', fill=False)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.add_artist(circle)

        colors = ['r', 'g', 'b', 'c', 'm']  # Colors for the vectors

        # Plot the principal components as vectors and annotate them
        for i in range(top_n):
            component = components[i, :]
            # Normalize the component for better visualization
            component = component / np.linalg.norm(component)

            # Get the top 3 words contributing to this component
            top_indices = np.argsort(np.abs(component))[-3:][::-1]
            top_words = [self.senses[idx] for idx in top_indices]

            plt.quiver(0, 0, component[0], component[1], angles='xy', scale_units='xy', scale=0.25, color=colors[i], label=f'PC{i+1}')
            plt.text(component[0], component[1], ', '.join(top_words), color=colors[i], fontsize=8, ha='center')

        # Formatting the plot
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.axhline(0, color='gray', linewidth=0.5)
        plt.axvline(0, color='gray', linewidth=0.5)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Top 5 Principal Components in Circular Plot')
        plt.legend()
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def pc_analyze_plot(self):
        """
        Static method to plot the principal components for a given word, with cmdline args
        """
        
        # Dictionary to hold words and their weighted magnitudes for each principal component
        pc_words = {}

        # Analyze the first 5 principal components
        for i in range(5):
            pc = self.VT[i]
            singular_value = self.S[i]
            weighted_pc = pc * singular_value  # Weight by the singular value
            sorted_indices = np.argsort(np.abs(weighted_pc))[::-1]
            top_words = [(self.senses[idx], np.abs(weighted_pc[idx])) for idx in sorted_indices[:3]]  # Top 4 words and their weighted magnitudes
            pc_words[f"PC{i+1}"] = top_words

        # Print the top words for each principal component
        for pc, words in pc_words.items():
            print(f"{pc}: {words}")

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 8))

        x_labels = list(pc_words.keys())
        x_positions = np.arange(len(x_labels))

        for pc_idx, (pc, words) in enumerate(pc_words.items()):
            heights = [magnitude for word, magnitude in words]
            
            bars = ax.bar(x_positions[pc_idx] + np.arange(len(words)) * 0.2, heights, width=0.2, label=pc)
            for bar, (word, magnitude) in zip(bars, words):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, word, ha='center', va='bottom', fontsize=8)

        ax.set_xlabel('Principal Components')
        ax.set_ylabel('Weighted Magnitude of Components')
        ax.set_title('Top 4 Words in Each Principal Component (Weighted by Singular Values)')
        ax.set_xticks(x_positions + 0.3)
        ax.set_xticklabels(x_labels)
        ax.legend()

        plt.show()

    def pca_3d_space_plot(self):
        """
        Plot the PCA in 3d space across the first 3 components
        """
        X = self.matrix

        # Perform PCA to reduce to 3 dimensions
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X.T)

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot of the 3D PCA results
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=0.5)

        # Annotate points with word labels
        for i, word in enumerate(self.senses):
            ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], word, fontsize=8, alpha=0.75)

        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.set_title('3D PCA of Word Embeddings')
        plt.show()
        

    
    def calculate_spectral_diversity(self, mode = "normal"):
        """
        Calculate the spectral diversity of the matrix
        Defined as the difference between the 0th, nth singular values for n singular values in a given SVD of the embeddings matrix.
        Author: Ella Qiawen Liu
        """
        if self.matrix.ndim < 2:  # ensure the array is at least two-dimensional
            return np.nan
        try:
            if mode == "normal":
                # Calculate the range of singular values
                diversity = np.ptp(self.S)
                return diversity
            elif mode == "absolute":
                # Calculate the cumulative variance explained by singular values
                explained_variance = self.pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)

                n = np.argmax(cumulative_variance >= 0.90) + 1

                if n < len(self.S):
                    diversity = self.S[0] - self.S[n]
                else:
                    diversity = self.S[0] - self.S[-1]
                return diversity
        except np.linalg.LinAlgError:
            return np.nan


    def count_senses(self, mode = "increment"):
        """
        Count the number of significant principal components(where proportion of explained variance > 0.01)
        
        Args:
            mode (str): The mode for counting senses. "increment" for explained variance > 0.01,
                        "absolute" for the top 90% of cumulative variance.

            Returns:
            int: The number of significant principal components.
        """
        if self.matrix.ndim < 2:
            return 0
        try:
            explained_variance = self.pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)

            if mode == "increment":
                differences = np.diff(cumulative_variance)
                concentration_index = np.argmax(differences < 0.01) + 1 
                return concentration_index

            elif mode == "absolute":
                concentration_index = np.argmax(cumulative_variance >= 0.90) + 1
                return concentration_index
        except:
            return 0
        

        
    def __pca_compute(self):
        """
        Private method to compute PCA and explained variance of components
        """
        pca = PCA()
        pca.fit(self.matrix)
        
        return pca
        

    def plot_scree(self):
        """
        Because the n_components will be equal to the number of cols in the matrix without dim. reduction, let's choose a threshold k 
        at which to cut off the number of components we include based on a point at which additional explained 
        variance of a given component q < 1% of the cumulative variance
        """
        explained_variance = self.pca.explained_variance_ratio_
        cum_variance = np.cumsum(explained_variance)

        differences = np.diff(cum_variance)
        elbow_index = np.argmax(differences < 0.01) + 1  # Adjust threshold as needed

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cum_variance) + 1), cum_variance, marker='o', linestyle='--')
        plt.axvline(x=elbow_index, color='r', linestyle='--')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by Number of Principal Components')
        plt.grid(True)
        plt.show()
        print(elbow_index)
        
        
    def calculate_eigenvalue_entropy(self):
        """
        Calculate the entropy of the eigenvalues derived from the singular values of the matrix.
        """
        if self.matrix.ndim < 2:
            return np.nan
        try:
            _, singular_values, _ = np.linalg.svd(self.matrix)
            eigenvalues = singular_values**2
            # Normalize eigenvalues to sum to 1
            total = np.sum(eigenvalues)
            if total > 0:
                normalized_eigenvalues = eigenvalues / total
                # Calculate entropy
                entropy = -np.sum(normalized_eigenvalues * np.log(normalized_eigenvalues + np.finfo(float).eps))
                return entropy
            else:
                return np.nan
        except:
            return np.nan
        
    def calculate_condition_number(self):
        """
        Calculate the condition number of the matrix using its singular values.

        In this case, the condition number is the ratio of the max/min non-singular values of the eigenvector matrix.

        """
        if self.matrix.ndim < 2:
            return np.nan
        try:
            if self.S.size == 0:
                return np.nan
            max_s = np.max(self.S)
            min_s = np.min(self.S[self.S > np.finfo(float).eps])
            return max_s / min_s if min_s > 0 else np.nan
        except:
            return np.nan


def run_semantic_compiler(word, decade, output_dir="./results"):
    """
    Runs the EmbeddingsMatrix analysis and outputs a CSV with key metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    matrix_dir = os.path.join('./staging', word, decade)
    matrix = EmbeddingsMatrix(matrix_dir, word)

    # Construct results row
    result = {
        "word": word,
        "decade": decade,
        "n_senses": matrix.n_senses,
        "spectral_diversity": matrix.spectral_diversity,
        "entropy": matrix.entropy,
        "condition_number": matrix.condition_number,
    }

    df_out = pd.DataFrame([result])
    out_path = os.path.join(output_dir, f"{word}_{decade}_semantic_metrics.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Saved semantic metrics to {out_path}")
    return df_out


# if __name__ == '__main__':
   
#     parser = argparse.ArgumentParser(description='Process word path and word name.')
#     parser.add_argument('word_name', type=str, help='The name of the word to look up in the embeddings')
#     parser.add_argument('decade', type=str, help='decade to extract')

#     args = parser.parse_args()   

#     word_dir =  f'{WORD_BASE_DIR}/{args.word_name}/{args.decade}'

#     matrix = EmbeddingsMatrix(word_dir, args.word_name)

#     matrix.pc_analyze_plot()

#     print("n senses:")
#     print(matrix.n_senses)
#     print("spectral diversity:")
#     print(matrix.spectral_diversity)
#     print("entropy:")
#     print(matrix.entropy)

#     matrix.plot_scree()
#     matrix.pca_3d_space_plot()

#     matrix.plot_pc_directions()