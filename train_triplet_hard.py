import torch
import os
import matplotlib.pyplot as plt
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

# SUBSET
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
# TRAIN
# ------------------------------
losses = []
os.makedirs("graphs", exist_ok=True)

for epoch in range(5):

    total_loss = 0   #  IMPORTANT

    for images, labels in loader:

        embeddings = model(images)

        loss = batch_hard_mining(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_loss = total_loss / len(loader)
    losses.append(epoch_loss)

    print("epoch:", epoch, "loss:", epoch_loss)

plt.plot(losses, marker='o')
plt.title("Triplet Hard Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("graphs/loss_triplet_hard.png")
plt.close()


# ------------------------------
# SAVE
# ------------------------------
torch.save(model.state_dict(), "weights/triplet_hard_model.pth")

print("✅ Triplet Hard training done!")