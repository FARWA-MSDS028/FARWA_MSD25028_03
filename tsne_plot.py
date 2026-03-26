import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(embeddings, labels, title="t-SNE", save_path="graphs/tsne_plot.png"):

    # Ensure folder exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print(f"\n🔍 Running t-SNE for: {title}")
    print(f"Total samples: {len(embeddings)}")

    # ✅ FIXED: use max_iter instead of n_iter
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        random_state=42,
        max_iter=1000
    )

    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    scatter = plt.scatter(
        embeddings_2d[:, 0],
        embeddings_2d[:, 1],
        c=labels,
        cmap="tab10",
        s=6
    )

    plt.colorbar(scatter)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"✅ Saved: {save_path}")


# -------------------------------------------------
# 🔴 MAIN (SO IT RUNS DIRECTLY)
# -------------------------------------------------
if __name__ == "__main__":

    print("Loading embeddings...")

    emb_path = "embeddings/test_embeddings.npy"
    lbl_path = "embeddings/test_labels.npy"

    if not os.path.exists(emb_path) or not os.path.exists(lbl_path):
        print("❌ Embedding files not found!")
        print("Run save_embeddings.py first")
    else:
        embeddings = np.load(emb_path)
        labels = np.load(lbl_path)

        plot_tsne(
            embeddings,
            labels,
            title="Test Embeddings",
            save_path="graphs/tsne_plot.png"
        )