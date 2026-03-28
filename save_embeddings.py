import torch
import numpy as np
import os

from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from model import EmbeddingNet
from embedding_utils import save_embeddings 

# -------------------------------------------------
# 🔴 FUNCTION: SAVE EMBEDDINGS
# -------------------------------------------------
def save_embeddings(dataset_path, model_path, save_prefix):
    """
    Generate and save embeddings + labels

    dataset_path: path to dataset (e.g., dataset/test)
    model_path: trained model weights
    save_prefix: output prefix (e.g., embeddings/test)
    """

    # ------------------------------
    # CHECK PATHS
    # ------------------------------
    if not os.path.exists(dataset_path):
        print("❌ Dataset path not found!")
        return

    if not os.path.exists(model_path):
        print("❌ Model not found!")
        return

    os.makedirs("embeddings", exist_ok=True)

    # ------------------------------
    # DEVICE
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # TRANSFORM (must match training)
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((32,32)),   # same as training
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # ------------------------------
    # DATASET
    # ------------------------------
    dataset = ImageFolder(dataset_path, transform=transform)

    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # ------------------------------
    # MODEL
    # ------------------------------
    model = EmbeddingNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # ------------------------------
    # GENERATE EMBEDDINGS
    # ------------------------------
    embeddings = []
    labels = []

    print("⚙️ Generating embeddings...")

    with torch.no_grad():
        for i, (images, lbls) in enumerate(loader):

            if i % 10 == 0:
                print(f"Processing batch {i}/{len(loader)}")

            images = images.to(device)

            emb = model(images)

            embeddings.append(emb.cpu())
            labels.append(lbls)

    # Combine
    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    # ------------------------------
    # SAVE FILES
    # ------------------------------
    np.save(save_prefix + "_embeddings.npy", embeddings.numpy())
    np.save(save_prefix + "_labels.npy", labels.numpy())

    print("\n✅ Embeddings saved successfully!")
    print("Shape:", embeddings.shape)


# -------------------------------------------------
# 🔴 MAIN
# -------------------------------------------------
if __name__ == "__main__":

    save_embeddings(
        dataset_path="dataset/test",
        model_path="weights/best_model.pth",
        save_prefix="embeddings/test"
    )