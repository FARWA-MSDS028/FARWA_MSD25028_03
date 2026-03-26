# #Contrastive Dataset
# import random
# from torch.utils.data import Dataset


# class ContrastiveDataset(Dataset):

#     def __init__(self, dataset):

#         self.dataset = dataset

#         self.labels = [label for _, label in dataset]

#         self.label_to_indices = {}

#         for idx, label in enumerate(self.labels):

#             if label not in self.label_to_indices:
#                 self.label_to_indices[label] = []

#             self.label_to_indices[label].append(idx)


#     def __getitem__(self, index):

#         img1, label1 = self.dataset[index]

#         if random.random() < 0.5:

#             label = 1

#             idx2 = index

#             while idx2 == index:

#                 idx2 = random.choice(self.label_to_indices[label1])

#         else:

#             label = 0

#             label2 = random.choice(list(self.label_to_indices.keys()))

#             while label2 == label1:

#                 label2 = random.choice(list(self.label_to_indices.keys()))

#             idx2 = random.choice(self.label_to_indices[label2])

#         img2, _ = self.dataset[idx2]

#         return img1, img2, label


#     def __len__(self):

#         return len(self.dataset)


# #Triplet Dataset
# class TripletDataset(Dataset):

#     def __init__(self, dataset):

#         self.dataset = dataset

#         self.labels = [label for _, label in dataset]

#         self.label_to_indices = {}

#         for idx, label in enumerate(self.labels):

#             if label not in self.label_to_indices:

#                 self.label_to_indices[label] = []

#             self.label_to_indices[label].append(idx)


#     def __getitem__(self, index):

#         anchor, label_anchor = self.dataset[index]

#         pos_idx = index

#         while pos_idx == index:

#             pos_idx = random.choice(self.label_to_indices[label_anchor])

#         positive, _ = self.dataset[pos_idx]


#         neg_label = random.choice(list(self.label_to_indices.keys()))

#         while neg_label == label_anchor:

#             neg_label = random.choice(list(self.label_to_indices.keys()))

#         neg_idx = random.choice(self.label_to_indices[neg_label])

#         negative, _ = self.dataset[neg_idx]

#         return anchor, positive, negative


#     def __len__(self):

#         return len(self.dataset)











































# if __name__ == "__main__":

#     from torchvision import transforms

#     transform = transforms.Compose([
#         transforms.Resize((224,224)),
#         transforms.ToTensor()
#     ])

#     dataset = ContrastiveDataset("dataset/train", transform)

#     print("Total samples:", len(dataset))

#     img1, img2, label = dataset[0]

#     print("Image1 shape:", img1.shape)
#     print("Image2 shape:", img2.shape)
#     print("Label:", label)

#     for i in range(20):
#         _, _, label = dataset[i]
#         print("Sample", i, "Label:", label)

# ............
#Contrastive Dataset
import random
from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):

    def __init__(self, dataset):
        # FIX: This expects a dataset, not a string
        self.dataset = dataset

        self.labels = [label for _, label in dataset]

        self.label_to_indices = {}

        for idx, label in enumerate(self.labels):

            if label not in self.label_to_indices:
                self.label_to_indices[label] = []

            self.label_to_indices[label].append(idx)

        # FIX: Add valid indices to avoid errors
        self.valid_indices = []
        for idx, label in enumerate(self.labels):
            if len(self.label_to_indices[label]) >= 2:
                self.valid_indices.append(idx)

    def __getitem__(self, index):
        # FIX: Use valid indices if provided, otherwise use index
        if hasattr(self, 'valid_indices') and len(self.valid_indices) > 0:
            actual_idx = self.valid_indices[index % len(self.valid_indices)]
        else:
            actual_idx = index
            
        img1, label1 = self.dataset[actual_idx]

        if random.random() < 0.5:
            label = 1
            # FIX: Ensure we get a different image for positive pair
            idx2 = actual_idx
            same_class_indices = self.label_to_indices[label1]
            if len(same_class_indices) > 1:
                while idx2 == actual_idx:
                    idx2 = random.choice(same_class_indices)
            else:
                idx2 = actual_idx

        else:
            label = 0
            # FIX: Ensure we get a different class for negative pair
            other_classes = [c for c in self.label_to_indices.keys() if c != label1]
            if other_classes:
                label2 = random.choice(other_classes)
                idx2 = random.choice(self.label_to_indices[label2])
            else:
                # Fallback to positive pair if no other class
                label = 1
                idx2 = actual_idx

        img2, _ = self.dataset[idx2]

        return img1, img2, label

    def __len__(self):
        # FIX: Return valid indices length if available
        if hasattr(self, 'valid_indices') and len(self.valid_indices) > 0:
            return len(self.valid_indices)
        return len(self.dataset)


#Triplet Dataset
class TripletDataset(Dataset):

    def __init__(self, dataset):
        # FIX: This expects a dataset, not a string
        self.dataset = dataset

        self.labels = [label for _, label in dataset]

        self.label_to_indices = {}

        for idx, label in enumerate(self.labels):

            if label not in self.label_to_indices:

                self.label_to_indices[label] = []

            self.label_to_indices[label].append(idx)

        # FIX: Only use anchors from classes with at least 2 images
        self.valid_anchors = []
        for idx, label in enumerate(self.labels):
            if len(self.label_to_indices[label]) >= 2:
                self.valid_anchors.append(idx)

    def __getitem__(self, index):
        # FIX: Use valid anchors
        anchor_idx = self.valid_anchors[index % len(self.valid_anchors)]
        anchor, label_anchor = self.dataset[anchor_idx]

        # FIX: Get different image for positive
        pos_idx = anchor_idx
        same_class_indices = [i for i in self.label_to_indices[label_anchor] if i != anchor_idx]
        if same_class_indices:
            pos_idx = random.choice(same_class_indices)
        positive, _ = self.dataset[pos_idx]

        # FIX: Get different class for negative
        other_classes = [c for c in self.label_to_indices.keys() if c != label_anchor]
        if other_classes:
            neg_label = random.choice(other_classes)
            neg_idx = random.choice(self.label_to_indices[neg_label])
        else:
            # Fallback - use same class different image
            neg_idx = random.choice(same_class_indices) if same_class_indices else anchor_idx

        negative, _ = self.dataset[neg_idx]

        return anchor, positive, negative

    def __len__(self):
        # FIX: Return valid anchors length
        return len(self.valid_anchors)


# FIX: Fix the test code at the bottom
if __name__ == "__main__":
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    import os

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # FIX: Load actual dataset, not string
    if os.path.exists("dataset/train"):
        raw_dataset = ImageFolder("dataset/train", transform=transform)
        
        # Test ContrastiveDataset
        contrastive_dataset = ContrastiveDataset(raw_dataset)
        print("ContrastiveDataset samples:", len(contrastive_dataset))

        img1, img2, label = contrastive_dataset[0]
        print("Image1 shape:", img1.shape)
        print("Image2 shape:", img2.shape)
        print("Label:", label)

        # Test TripletDataset
        triplet_dataset = TripletDataset(raw_dataset)
        print("\nTripletDataset samples:", len(triplet_dataset))
        
        anchor, positive, negative = triplet_dataset[0]
        print("Anchor shape:", anchor.shape)
        print("Positive shape:", positive.shape)
        print("Negative shape:", negative.shape)
        
        print("\n✅ Dataset tests passed!")
    else:
        print("Please run split_dataset.py first to create dataset/train folder")
