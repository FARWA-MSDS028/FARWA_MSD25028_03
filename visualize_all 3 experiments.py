# import torch
# import numpy as np
# import os
# import matplotlib.pyplot as plt

# from torchvision import transforms
# from torchvision.datasets import ImageFolder
# from torch.utils.data import DataLoader
# from sklearn.manifold import TSNE

# from model import EmbeddingNet


# # -------------------------------------------------
# # FUNCTION: Generate embeddings from model
# # -------------------------------------------------
# def get_embeddings(model_path, loader, device):
#     """
#     Loads model and generates embeddings, labels, and images

#     Args:
#         model_path: path to saved model weights
#         loader: dataloader for dataset
#         device: cpu or cuda

#     Returns:
#         embeddings, labels, images
#     """

#     # Load model
#     model = EmbeddingNet()
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = model.to(device)
#     model.eval()

#     all_embeddings = []
#     all_labels = []
#     all_images = []

#     # Disable gradient (inference mode)
#     with torch.no_grad():
#         for images, labels in loader:

#             images = images.to(device)

#             # Forward pass → embeddings
#             emb = model(images)

#             # Store results
#             all_embeddings.append(emb.cpu())
#             all_labels.append(labels)
#             all_images.append(images.cpu())

#     # Combine all batches
#     embeddings = torch.cat(all_embeddings)
#     labels = torch.cat(all_labels)
#     images = torch.cat(all_images)

#     return embeddings, labels, images


# # -------------------------------------------------
# # FUNCTION: t-SNE Visualization
# # -------------------------------------------------
# def plot_tsne(embeddings, labels, name):
#     """
#     Creates 2D t-SNE plot of embeddings

#     Args:
#         embeddings: (N, embedding_dim)
#         labels: (N,)
#         name: experiment name (used in filename)
#     """

#     os.makedirs("graphs", exist_ok=True)

#     print(f"🔍 Running t-SNE for {name}...")

#     # Apply t-SNE
#     tsne = TSNE(
#         n_components=2,
#         perplexity=30,
#         random_state=42,
#         max_iter=1000   # IMPORTANT: new sklearn uses max_iter
#     )

#     emb_2d = tsne.fit_transform(embeddings.numpy())

#     # Plot
#     plt.figure(figsize=(8, 6))
#     plt.scatter(
#         emb_2d[:, 0],
#         emb_2d[:, 1],
#         c=labels.numpy(),
#         s=5,
#         cmap="tab10"
#     )

#     plt.title(f"{name} t-SNE")

#     save_path = f"graphs/tsne_{name}.png"

#     plt.savefig(save_path)
#     plt.close()

#     print(f"✅ Saved: {save_path}")


# # -------------------------------------------------
# # FUNCTION: Retrieval Visualization
# # -------------------------------------------------
# def show_retrieval(query_idx, embeddings, images, labels, name, k=5):
#     """
#     Shows query image + top-k nearest neighbors

#     Green = correct class
#     Red = wrong class
#     """

#     # Compute distances from query to all
#     distances = torch.cdist(
#         embeddings[query_idx].unsqueeze(0),
#         embeddings
#     ).squeeze(0)

#     # Remove self-match
#     distances[query_idx] = float("inf")

#     # Get top-k nearest neighbors
#     nearest = torch.topk(
#         distances,
#         k,
#         largest=False
#     ).indices

#     # Create plot grid
#     fig, axes = plt.subplots(1, k+1, figsize=(15, 3))

#     # Query image
#     axes[0].imshow(images[query_idx].permute(1, 2, 0))
#     axes[0].set_title("Query")
#     axes[0].axis("off")

#     query_label = labels[query_idx]

#     # Loop through neighbors
#     for i in range(k):

#         idx = nearest[i]

#         axes[i+1].imshow(images[idx].permute(1, 2, 0))

#         # Color logic
#         if labels[idx] == query_label:
#             color = "green"   # correct match
#         else:
#             color = "red"     # wrong match

#         axes[i+1].set_title(f"Top-{i+1}", color=color)
#         axes[i+1].axis("off")

#     # Save image
#     save_path = f"graphs/{name}_retrieval_{query_idx}.png"

#     plt.tight_layout()
#     plt.savefig(save_path)
#     plt.close()

#     print(f"✅ Saved: {save_path}")


# # -------------------------------------------------
# # MAIN FUNCTION
# # -------------------------------------------------
# if __name__ == "__main__":

#     # Select device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # Same transform used everywhere
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),
#         transforms.ToTensor()
#     ])

#     # Load TEST dataset
#     dataset = ImageFolder("dataset/test", transform=transform)

#     loader = DataLoader(
#         dataset,
#         batch_size=32,
#         shuffle=False
#     )

#     # 3 experiments
#     models = {
#         "contrastive": "weights/best_model.pth",
#         "triplet_random": "weights/triplet_random_model.pth",
#         "triplet_hard": "weights/triplet_hard_model.pth"
#     }

#     # Loop over experiments
#     for name, path in models.items():

#         print(f"\n🚀 Processing {name} model...")

#         # Generate embeddings
#         embeddings, labels, images = get_embeddings(path, loader, device)

#         # -------- t-SNE --------
#         plot_tsne(embeddings, labels, name)

#         # -------- Retrieval (10 queries) --------
#         for i in range(10):
#             show_retrieval(i, embeddings, images, labels, name)

#     print("\n🎉 ALL VISUALIZATIONS COMPLETED!") 


import torch
import numpy as np

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import EmbeddingNet


# -------------------------------------------------
# RECALL FUNCTION (your same logic)
# -------------------------------------------------
def recall_at_k(embeddings, labels, k=1):

    N = embeddings.size(0)

    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    correct = 0

    for i in range(N):

        distances = distance_matrix[i]
        distances[i] = float('inf')

        nearest = torch.topk(distances, k, largest=False).indices

        if (labels[nearest] == labels[i]).any():
            correct += 1

    return correct / N


# -------------------------------------------------
# GENERATE EMBEDDINGS
# -------------------------------------------------
def get_embeddings(model_path, loader, device):

    model = EmbeddingNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:

            images = images.to(device)

            emb = model(images)

            all_embeddings.append(emb.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    return embeddings, labels


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # DATASET (same for all)
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ])

    dataset = ImageFolder("dataset/test", transform=transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # ------------------------------
    # MODELS (3 experiments)
    # ------------------------------
    models = {
        "Contrastive": "weights/best_model.pth",
        "Triplet Random": "weights/triplet_random_model.pth",
        "Triplet Hard": "weights/triplet_hard_model.pth"
    }

    # ------------------------------
    # EVALUATION
    # ------------------------------
    print("\n🔍 Evaluating all models...\n")

    for name, path in models.items():

        print(f"📌 {name}")

        embeddings, labels = get_embeddings(path, loader, device)

        r1 = recall_at_k(embeddings, labels, k=1)
        r5 = recall_at_k(embeddings, labels, k=5)

        print(f"Recall@1: {r1:.4f}")
        print(f"Recall@5: {r5:.4f}\n")

    print("✅ Evaluation complete!")