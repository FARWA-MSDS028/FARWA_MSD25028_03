import torch
import numpy as np
import os


def save_embeddings(embeddings, labels, images, prefix="test"):
    """
    Save embeddings, labels, and images efficiently
    """

    os.makedirs("embeddings", exist_ok=True)

    # Save embeddings
    np.save(
        f"embeddings/{prefix}_embeddings.npy",
        embeddings.numpy()
    )

    # Save labels
    np.save(
        f"embeddings/{prefix}_labels.npy",
        labels.numpy()
    )

    # Save images tensor
    torch.save(
        images,
        f"embeddings/{prefix}_images.pt"
    )

    print("✅ Embeddings saved successfully!")


def load_embeddings(prefix="test"):
    """
    Load embeddings, labels, and images
    """

    embeddings = torch.tensor(
        np.load(f"embeddings/{prefix}_embeddings.npy")
    )

    labels = torch.tensor(
        np.load(f"embeddings/{prefix}_labels.npy")
    )

    images = torch.load(
        f"embeddings/{prefix}_images.pt"
    )

    return embeddings, labels, images