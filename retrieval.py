import torch
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from model import EmbeddingNet


# -------------------------------------------------
# 🔴 RECALL@K FUNCTION (KEEP AS PROVIDED ✔)
# -------------------------------------------------
def recall_at_k(embeddings, labels, k=1):
    """
    embeddings: tensor (N, embedding_dim)
    labels: tensor (N,)
    """

    N = embeddings.size(0)

    # Pairwise distance matrix
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)

    correct = 0

    for i in range(N):

        distances = distance_matrix[i]

        # Exclude self
        distances[i] = float('inf')

        # Top-K nearest
        nearest_indices = torch.topk(distances, k, largest=False).indices

        # Check match
        if (labels[nearest_indices] == labels[i]).any():
            correct += 1

    return correct / N
    return recall


# -------------------------------------------------
# 🔴 MAIN EXECUTION
# -------------------------------------------------
if __name__ == "__main__":

    print("🔍 Starting Retrieval Evaluation...\n")

    # ------------------------------
    # CHECK FILES
    # ------------------------------
    if not os.path.exists("dataset/test"):
        print("❌ dataset/test not found!")
        exit()

    if not os.path.exists("weights/best_model.pth"):
        print("❌ Model not found! Train first.")
        exit()

    # ------------------------------
    # DEVICE
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # ------------------------------
    # TRANSFORM (same as inference)
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((32,32)),   # MUST match training
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # ------------------------------
    # LOAD DATASET
    # ------------------------------
    print("📂 Loading test dataset...")

    test_dataset = ImageFolder("dataset/test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Total test images: {len(test_dataset)}\n")

    # ------------------------------
    # LOAD MODEL
    # ------------------------------
    print("🧠 Loading model...")

    model = EmbeddingNet()
    model.load_state_dict(torch.load("weights/best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # ------------------------------
    # GENERATE EMBEDDINGS
    # ------------------------------
    print("⚙️ Generating embeddings...")

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):

            if i % 10 == 0:
                print(f"Processing batch {i}/{len(test_loader)}")

            images = images.to(device)

            embeddings = model(images)

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels)

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)

    print("\n✅ Embeddings generated!")
    print("Embedding shape:", embeddings.shape)

    # ------------------------------
    # COMPUTE RECALL
    # ------------------------------
    print("\n📊 Computing Recall...")

    r1 = recall_at_k(embeddings, labels, k=1)
    r5 = recall_at_k(embeddings, labels, k=5)

    print("\n🎯 FINAL RESULTS:")
    print("Recall@1:", round(r1, 4))
    print("Recall@5:", round(r5, 4))