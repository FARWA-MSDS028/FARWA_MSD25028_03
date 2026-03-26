#
from train import train

if __name__ == "__main__":
    train()



#embeding
    from train import train
from embeddings import generate_embeddings

if __name__ == "__main__":
    
    train()

    generate_embeddings(
        "dataset/test",
        "saved_models/contrastive.pth",
        "embeddings/test"
    )