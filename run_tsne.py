import numpy as np
from tsne_plot import plot_tsne

emb = np.load("embeddings/test_embeddings.npy")
labels = np.load("embeddings/test_labels.npy")

plot_tsne(
    emb,
    labels,
    title="Test Embeddings",
    save_path="graphs/tsne_plot.png"
)