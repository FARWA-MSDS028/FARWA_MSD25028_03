import torch
import os

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import EmbeddingNet
from loss import batch_hard_mining


# ------------------------------
# SETUP
# ------------------------------
os.makedirs("weights", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------
# TRANSFORM
# ------------------------------
transform = transforms.Compose([
        transforms.Resize((32,32)), 
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

# ------------------------------
# DATASET (YOUR DATA)
# ------------------------------
dataset = ImageFolder("dataset/train", transform=transform)

# 🔥 SUBSET
dataset = Subset(dataset, range(1000))


loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)


# ------------------------------
# MODEL
# ------------------------------
model = EmbeddingNet().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------
# TRAINING
# ------------------------------
for epoch in range(3):

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        embeddings = model(images)

        loss = batch_hard_mining(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch:", epoch, "loss:", loss.item())


# ------------------------------
# SAVE
# ------------------------------
torch.save(model.state_dict(), "weights/triplet_hard_model.pth")

print("✅ Triplet Hard training done!")