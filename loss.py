# #Contrastive Loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class ContrastiveLoss(nn.Module):

#     def __init__(self, margin=1):

#         super().__init__()

#         self.margin = margin


#     def forward(self, emb1, emb2, label):

#         distance = F.pairwise_distance(emb1, emb2)

#         loss = torch.mean(

#             label * distance**2 +

#             (1-label) * torch.clamp(self.margin-distance, min=0)**2

#         )

#         return loss




# #Triplet Loss
# class TripletLoss(nn.Module):

#     def __init__(self, margin=1):

#         super().__init__()

#         self.margin = margin


#     def forward(self, anchor, positive, negative):

#         pos_dist = F.pairwise_distance(anchor, positive)

#         neg_dist = F.pairwise_distance(anchor, negative)

#         loss = torch.mean(

#             torch.clamp(pos_dist-neg_dist+self.margin, min=0)

#         )

#         return loss
    



# #Add Batch Hard Mining
# import torch
# import torch.nn.functional as F

# def batch_hard_mining(embeddings, labels, margin=0.2):
#     """
#     embeddings: tensor of shape (batch_size, embedding_dim)
#     labels: tensor of shape (batch_size)
    
#     Returns:
#     triplet loss computed using hardest positives and negatives
#     """

#     batch_size = embeddings.size(0)
    
#     # Compute pairwise distance matrix
#     distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    
#     loss = 0.0
#     valid_triplets = 0

#     for i in range(batch_size):
        
#         anchor_label = labels[i]
        
#         # mask for positives (same class)
#         positive_mask = (labels == anchor_label)
#         positive_mask[i] = False  # remove anchor itself
        
#         # mask for negatives (different class)
#         negative_mask = (labels != anchor_label)
        
#         if positive_mask.sum() == 0:
#             continue
        
#         # hardest positive = max distance among positives
#         hardest_positive_dist = distance_matrix[i][positive_mask].max()
        
#         # hardest negative = min distance among negatives
#         hardest_negative_dist = distance_matrix[i][negative_mask].min()
        
#         # compute triplet loss
#         triplet_loss = F.relu(
#             hardest_positive_dist - hardest_negative_dist + margin
#         )
        
#         loss += triplet_loss
#         valid_triplets += 1

#     if valid_triplets == 0:
#         return torch.tensor(0.0, requires_grad=True)

#     loss = loss / valid_triplets
    
#     return loss
# .....................
#Contrastive Loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=1):
        super().__init__()

        self.margin = margin

    def forward(self, emb1, emb2, label):
        # FIX: Ensure label is float
        label = label.float()

        distance = F.pairwise_distance(emb1, emb2)

        loss = torch.mean(

            label * distance**2 +

            (1-label) * torch.clamp(self.margin-distance, min=0)**2

        )

        return loss




#Triplet Loss
class TripletLoss(nn.Module):

    # FIX: Change margin from 1 to 0.2 as per assignment
    def __init__(self, margin=0.2):
        super().__init__()

        self.margin = margin

    def forward(self, anchor, positive, negative):

        pos_dist = F.pairwise_distance(anchor, positive)

        neg_dist = F.pairwise_distance(anchor, negative)

        loss = torch.mean(

            torch.clamp(pos_dist - neg_dist + self.margin, min=0)

        )

        return loss
    


#Batch Hard Mining
def batch_hard_mining(embeddings, labels, margin=0.2):
    """
    embeddings: tensor of shape (batch_size, embedding_dim)
    labels: tensor of shape (batch_size)
    
    Returns:
    triplet loss computed using hardest positives and negatives
    """

    batch_size = embeddings.size(0)
    
    # Compute pairwise distance matrix
    distance_matrix = torch.cdist(embeddings, embeddings, p=2)
    
    # FIX: Create masks using broadcasting for efficiency
    labels_expanded = labels.unsqueeze(1)
    positive_mask = (labels_expanded == labels_expanded.t()).float()
    negative_mask = 1 - positive_mask
    
    # FIX: Remove self from positive mask
    positive_mask.fill_diagonal_(0)
    
    # FIX: Find hardest positive and negative efficiently
    positive_distances = distance_matrix * positive_mask
    positive_distances[positive_mask == 0] = -float('inf')
    hardest_positive_dist, _ = positive_distances.max(dim=1)
    
    negative_distances = distance_matrix * negative_mask
    negative_distances[negative_mask == 0] = float('inf')
    hardest_negative_dist, _ = negative_distances.min(dim=1)
    
    # FIX: Filter valid triplets
    valid_triplets = (hardest_positive_dist != -float('inf')) & (hardest_negative_dist != float('inf'))
    
    if valid_triplets.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=embeddings.device)
    
    # FIX: Compute loss only on valid triplets
    loss = torch.clamp(
        hardest_positive_dist[valid_triplets] - hardest_negative_dist[valid_triplets] + margin,
        min=0
    )
    
    return loss.mean()