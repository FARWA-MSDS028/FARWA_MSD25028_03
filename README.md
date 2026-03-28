# Deep Metric Learning for Image Retrieval using CIFAR-10

Name: Farwa
Roll Number: MSD25028  

## Project Description

In this assignment, I implemented and compared three different deep metric learning methods for image retrieval using the CIFAR-10 dataset. Instead of doing normal classification, the goal of this project was to learn embeddings where similar images stay close together and different images stay far apart.

The three methods used in this project are:

1. Contrastive Loss
2. Triplet Loss (Random Triplets)
3. Triplet Loss (Hard Negative Mining)

All models were trained using the same architecture so that the comparison between methods remains fair.

---

## Dataset

I used the CIFAR-10 dataset for this assignment. It contains 10 image classes:

Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
,The CIFAR-10 dataset contains 60,000 color images belonging to 10 object classes
The dataset was used to generate pairs for contrastive learning and triplets for triplet learning.
Images were resized, normalized, and used to generate pairs and triplets for metric learning experiments
---

## Model Architecture

For feature extraction, I used ResNet-50 as a backbone network. The final classification layer was removed and replaced with a 128-dimensional embedding layer.

These embeddings were then used to measure similarity between images.

The same architecture was used for all three experiments.

---

## Experiments Performed

Three experiments were completed:

Experiment 1:
Contrastive Loss with random image pairs

Experiment 2:
Triplet Loss with random triplets

Experiment 3:
Triplet Loss with hard negative mining

---

## Evaluation Method

To evaluate performance, I used:

Recall@1  
Recall@5  

These metrics check whether the correct class appears among the nearest retrieved images.

Results are shown below:

Contrastive Loss
Recall@1 = 0.1414
Recall@5 = 0.3005

Triplet Loss (Random)
Recall@1 = 0.1626
Recall@5 = 0.2984

Triplet Loss (Hard Negative Mining)
Recall@1 = 0.1499
Recall@5 = 0.2885

Triplet Loss with random sampling performed slightly better in Recall@1 compared to the other methods.

---

## Visualization

To understand how embeddings are distributed, I generated t-SNE plots for all three models.

Also, retrieval visualization was done using 10 query images. For each query image, the top-5 nearest images were displayed. Correct matches appear in green and incorrect matches appear in red.

These results help show how well the embedding space separates different classes.

---

## Files in this Project

Main files included in this project:

train_contrastive.py  
train_triplet_random.py  
train_triplet_hard.py  
visualize_experiments.py  
evaluate_recall.py  

Graphs are saved inside the graphs folder.

---

## How to Run the Project

First install dependencies:

pip install -r requirements.txt

Then run training scripts:

python train_contrastive.py  
python train_triplet_random.py  
python train_triplet_hard.py  

After training, run:

python visualize_experiments.py

This will generate Recall results, t-SNE plots, and retrieval visualizations.
Retrieval performance was evaluated using:

- Recall@1
- Recall@5 
---

## Conclusion

In this assignment, I compared three deep metric learning approaches for image retrieval. The results show that Triplet Loss with random sampling performed slightly better for Recall@1, while Contrastive Loss gave competitive results for Recall@5.

Overall, all three methods successfully learned meaningful embeddings for similarity-based image retrieval.
AI Tool Usage:
ChatGPT was used for conceptual guidance, debugging, and report structuring.