import torch
import os

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

from model import EmbeddingNet
from dataset import TripletDataset
from loss import TripletLoss


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

triplet_dataset = TripletDataset(dataset)

# 🔥 SUBSET
triplet_dataset = Subset(triplet_dataset, range(1000))


loader = DataLoader(
    triplet_dataset,
    batch_size=32,
    shuffle=True
)


# ------------------------------
# MODEL
# ------------------------------
model = EmbeddingNet().to(device)

criterion = TripletLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# ------------------------------
# TRAINING
# ------------------------------
for epoch in range(3):

    for anchor, positive, negative in loader:

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("epoch:", epoch, "loss:", loss.item())


# ------------------------------
# SAVE
# ------------------------------
torch.save(model.state_dict(), "weights/triplet_random_model.pth")

print("✅ Triplet Random training done!")