import torch
import os
import matplotlib.pyplot as plt
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

#  SUBSET
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
losses = []
os.makedirs("graphs", exist_ok=True)

for epoch in range(5):

    total_loss = 0  

    for anchor, positive, negative in loader:

        emb_a = model(anchor)
        emb_p = model(positive)
        emb_n = model(negative)

        loss = criterion(emb_a, emb_p, emb_n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(loader)
    losses.append(epoch_loss)

    print("epoch:", epoch, "loss:", epoch_loss)

plt.plot(losses, marker='o')
plt.title("Triplet Random Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("graphs/loss_triplet_random.png")
plt.close()


# ------------------------------
# SAVE
# ------------------------------
torch.save(model.state_dict(), "weights/triplet_random_model.pth")

print("✅ Triplet Random training done!")