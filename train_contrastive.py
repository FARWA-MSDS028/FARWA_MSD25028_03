import torch
import os

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import EmbeddingNet
from dataset import ContrastiveDataset
from loss import ContrastiveLoss


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

contrastive_dataset = ContrastiveDataset(dataset)

# 🔥 SUBSET (FAST TRAINING)
contrastive_dataset = Subset(contrastive_dataset, range(1000))


loader = DataLoader(
    contrastive_dataset,
    batch_size=32,
    shuffle=True
)


# ------------------------------
# MODEL
# ------------------------------
model = EmbeddingNet().to(device)

criterion = ContrastiveLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------
# TRAINING
# ------------------------------
for epoch in range(3):

    for img1, img2, label in loader:

        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.float().to(device)

        emb1 = model(img1)
        emb2 = model(img2)

        loss = criterion(emb1, emb2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch:", epoch, "loss:", loss.item())


# ------------------------------
# SAVE
# ------------------------------
torch.save(model.state_dict(), "weights/contrastive_model.pth")

print("✅ Contrastive training done!")