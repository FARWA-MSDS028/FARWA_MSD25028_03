
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import os

from model import EmbeddingNet


# ------------------------------
# SAME TRANSFORM AS TRAINING
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((32,32)),   # must match training
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def infer_image(model_path, image_path):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------------
    # LOAD MODEL
    # ------------------------------
    model = EmbeddingNet()

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    # ------------------------------
    # LOAD IMAGE
    # ------------------------------
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)

    # ------------------------------
    # INFERENCE
    # ------------------------------
    with torch.no_grad():
        embedding = model(image)

    return embedding.cpu().numpy()


# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":

    model_path = "weights/best_model.pth"

    # ✅ FIX: correct path to your test image
    image_path = "dataset/test/accordion/image_0021.jpg"

    if not os.path.exists(model_path):
        print(" Model not found!")
        print("Run train.py first")

    elif not os.path.exists(image_path):
        print(" Image not found!")
        print("Check your image path inside dataset/test/")
    
    else:
        emb = infer_image(model_path, image_path)

        print("✅ Inference successful!")
        print("Embedding shape:", emb.shape)
        print("Embedding norm:", np.linalg.norm(emb))