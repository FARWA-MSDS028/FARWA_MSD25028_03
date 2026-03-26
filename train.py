# import torch
# import matplotlib.pyplot as plt
# import os

# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms
# from torchvision.datasets import ImageFolder

# from model import EmbeddingNet
# from dataset import ContrastiveDataset
# from loss import ContrastiveLoss


# # Transform
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor()
# ])

# # Load dataset using ImageFolder
# raw_dataset = ImageFolder("dataset/train", transform=transform)

# # Wrap in ContrastiveDataset
# dataset = ContrastiveDataset(raw_dataset)

# # Split dataset
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size

# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32)

# # Model
# model = EmbeddingNet()

# # Loss and optimizer
# criterion = ContrastiveLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# # Early stopping parameters
# best_val_loss = float("inf")
# patience = 5
# counter = 0

# train_losses = []
# val_losses = []

# # Training loop
# for epoch in range(20):
#     model.train()
#     train_loss = 0

#     for img1, img2, label in train_loader:
#         emb1 = model(img1)
#         emb2 = model(img2)

#         loss = criterion(emb1, emb2, label.float())

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item()

#     train_loss /= len(train_loader)
#     train_losses.append(train_loss)

#     # Validation
#     model.eval()
#     val_loss = 0

#     with torch.no_grad():
#         for img1, img2, label in val_loader:
#             emb1 = model(img1)
#             emb2 = model(img2)
#             loss = criterion(emb1, emb2, label.float())
#             val_loss += loss.item()

#     val_loss /= len(val_loader)
#     val_losses.append(val_loss)

#     print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#     # Early stopping
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         counter = 0

#         torch.save(model.state_dict(), "weights/best_model.pth")
#         print("Model improved and saved")

#     else:
#         counter += 1
#         if counter >= patience:
#             print("Early stopping triggered")
#             break

# # Plot loss graph
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Validation Loss")
# plt.legend()
# plt.title("Training vs Validation Loss")
# plt.savefig("graphs/loss_plot.png")
# plt.show()
# ..................
import torch
import matplotlib.pyplot as plt
import os

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import EmbeddingNet
from dataset import ContrastiveDataset
from loss import ContrastiveLoss


def main():

    # ------------------------------
    # FAST TRANSFORM
    # ------------------------------
    transform = transforms.Compose([
        transforms.Resize((32,32)),   # faster training
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # ------------------------------
    # LOAD DATASET
    # ------------------------------
    if not os.path.exists("dataset/train"):
        print("Error: dataset/train not found!")
        return

    raw_dataset = ImageFolder("dataset/train", transform=transform)
    dataset = ContrastiveDataset(raw_dataset)

    # ------------------------------
    # SPLIT DATA
    # ------------------------------
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 🔥 SMALL SUBSET (for fast testing)
    train_dataset = Subset(train_dataset, range(1000))
    val_dataset = Subset(val_dataset, range(200))

    # ------------------------------
    # DATALOADER
    # ------------------------------
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, num_workers=0)

    # ------------------------------
    # MODEL
    # ------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmbeddingNet().to(device)

    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------
    # SETUP
    # ------------------------------
    os.makedirs("weights", exist_ok=True)
    os.makedirs("graphs", exist_ok=True)

    best_val_loss = float("inf")
    patience = 3
    counter = 0

    train_losses = []
    val_losses = []

    print("Starting training...")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # ------------------------------
    # TRAINING LOOP
    # ------------------------------
    for epoch in range(3):   # fast training

        model.train()
        train_loss = 0

        for i, (img1, img2, label) in enumerate(train_loader):

            # progress print
            if i % 20 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(train_loader)}")

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.float().to(device)

            emb1 = model(img1)
            emb2 = model(img2)

            loss = criterion(emb1, emb2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # ------------------------------
        # VALIDATION
        # ------------------------------
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for img1, img2, label in val_loader:

                img1 = img1.to(device)
                img2 = img2.to(device)
                label = label.float().to(device)

                emb1 = model(img1)
                emb2 = model(img2)

                val_loss += criterion(emb1, emb2, label).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"\nEpoch {epoch+1} DONE")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n")

        # ------------------------------
        # EARLY STOPPING
        # ------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0

            torch.save(model.state_dict(), "weights/best_model.pth")
            print("✅ Model improved and saved")

        else:
            counter += 1
            print(f"No improvement ({counter}/{patience})")

            if counter >= patience:
                print("⛔ Early stopping triggered")
                break

    # ------------------------------
    # PLOT LOSS
    # ------------------------------
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Val")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig("graphs/loss.png")

    print("\n✅ Training complete!")
    print("Model saved at: weights/best_model.pth")


# ------------------------------
# WINDOWS FIX (VERY IMPORTANT)
# ------------------------------
if __name__ == "__main__":
    main()