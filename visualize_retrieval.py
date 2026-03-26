import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def show_retrieval(query_idx, embeddings, images, labels, k=5):
    """
    Show Top-K retrieval for one query image
    """

    # ------------------------------
    # COMPUTE DISTANCE (Euclidean)
    # ------------------------------
    distances = torch.cdist(
        embeddings[query_idx].unsqueeze(0),
        embeddings
    ).squeeze(0)

    # Remove self-match
    distances[query_idx] = float("inf")

    # Get Top-K nearest neighbors
    nearest = torch.topk(
        distances,
        k,
        largest=False
    ).indices

    # ------------------------------
    # CREATE OUTPUT FOLDER
    # ------------------------------
    os.makedirs("graphs", exist_ok=True)

    # ------------------------------
    # PLOT GRID (Query + Top-K)
    # ------------------------------
    fig, axes = plt.subplots(1, k+1, figsize=(15, 3))

    # Query Image
    axes[0].imshow(images[query_idx].permute(1, 2, 0))
    axes[0].set_title("Query")
    axes[0].axis("off")

    query_label = labels[query_idx]

    # Top-K Images
    for i in range(k):

        idx = nearest[i]

        axes[i+1].imshow(images[idx].permute(1, 2, 0))

        retrieved_label = labels[idx]
        print(f"Top-{i+1} Label: {retrieved_label}")

        axes[i+1].imshow(images[idx].permute(1, 2, 0))

        # Green = correct, Red = wrong
        if retrieved_label == query_label:
            color = "green"
        else:
            color = "red"

        axes[i+1].set_title(f"Top-{i+1}", color=color)
        axes[i+1].axis("off")

    # Save each query result
    save_path = f"graphs/retrieval_{query_idx}.png"

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved: {save_path}")


# -------------------------------------------------
# 🔴 MAIN (RUN EVERYTHING HERE)
# -------------------------------------------------
if __name__ == "__main__":

    print("🔍 Loading embeddings...")

    # Load embeddings and labels
    embeddings = torch.tensor(
        np.load("embeddings/test_embeddings.npy")
    )

    labels = torch.tensor(
        np.load("embeddings/test_labels.npy")
    )

    # ------------------------------
    # LOAD IMAGES (generate if not exists)
    # ------------------------------
    if not os.path.exists("embeddings/test_images.pt"):

        print("⚙️ Creating image tensor...")

        from torchvision import transforms
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader

        transform = transforms.Compose([
            transforms.Resize((32,32)),
            transforms.ToTensor()
        ])

        dataset = ImageFolder("dataset/test", transform=transform)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        all_images = []

        for imgs, _ in loader:
            all_images.append(imgs)

        images = torch.cat(all_images)

        torch.save(images, "embeddings/test_images.pt")

        print("✅ Images saved!")

    else:
        print("✅ Loading saved images...")
        images = torch.load("embeddings/test_images.pt")

    # ------------------------------
    # GENERATE 10 RETRIEVAL RESULTS
    # ------------------------------
    print("📊 Generating retrieval results...")

    for i in range(10):

        show_retrieval(
            query_idx=i,
            embeddings=embeddings,
            images=images,
            labels=labels,
            k=5
        )

    print("\n✅ DONE! Check graphs/ folder")