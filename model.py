# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torchvision import models


# class EmbeddingNet(nn.Module):

#     def __init__(self, embedding_dim=128):
#         super().__init__()

#         resnet = models.resnet50(pretrained=True)

#         # Remove classifier layer
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])

#         # Projection head
#         self.fc = nn.Linear(2048, embedding_dim)

#     def forward(self, x):

#         x = self.backbone(x)

#         x = x.view(x.size(0), -1)

#         x = self.fc(x)

#         x = F.normalize(x, p=2, dim=1)

#         return x
    

# if __name__ == "__main__":

#     model = EmbeddingNet()

#     print(model)

#     # create dummy input (1 image, 3 channels, 224x224)
#     x = torch.randn(1, 3, 224, 224)

#     output = model(x)

#     print("Output shape:", output.shape)
# .................
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class EmbeddingNet(nn.Module):

    def __init__(self, embedding_dim=128):
        super().__init__()

        resnet = models.resnet50(pretrained=True)

        # Remove classifier layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Projection head - ADD BATCHNORM AND RELU
        self.fc = nn.Linear(2048, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)  # ADD THIS LINE
        self.relu = nn.ReLU(inplace=True)  # ADD THIS LINE

    def forward(self, x):

        x = self.backbone(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        x = self.bn(x)  # ADD THIS LINE
        x = self.relu(x)  # ADD THIS LINE

        x = F.normalize(x, p=2, dim=1)

        return x
    

if __name__ == "__main__":

    model = EmbeddingNet()

    print(model)

    # create dummy input (1 image, 3 channels, 224x224)
    x = torch.randn(1, 3, 224, 224)

    output = model(x)

    print("Output norm:", torch.norm(output, dim=1))  # ADD THIS LINE