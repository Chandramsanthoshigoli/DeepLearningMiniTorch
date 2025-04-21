# Mintorch Neural Network Classifier

This project implements a simple neural network classifier using a custom deep learning library called **Mintorch**, built from scratch to mimic core PyTorch functionalities. The goal is to classify points into different classes and visualize the loss and decision boundaries.

---

## 📊 Output Screenshots

### 1. Loss Graph

This graph shows how the model's loss changes over time (epochs). You can observe the learning process through the fluctuations and eventual convergence of the loss value.

![Loss Graph](output%20santhoshi.jpg)

---

### 2. Classification Decision Boundary

This plot shows how the model separates different classes based on the learned weights. Each colored region represents a predicted class, and the points indicate the data distribution.

![Classification Plot](output2%20santhoshi.jpg)

---

## 📂 Training Log

A detailed log of the training process is saved in [`training_log.csv`](training_log.csv), which records:

- Epochs (countdown style)
- Loss values per epoch
- Number of correct predictions per epoch

### Sample Log Preview

| Epoch | Loss     | Correct |
|-------|----------|---------|
| 500   | 0.721928 | 50      |
| 490   | 0.750417 | 50      |
| 480   | 0.780092 | 50      |
| ...   | ...      | ...     |

---

## ⚙️ How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Chandramsanthoshigoli/DeepLearningMiniTorch.git
   cd DeepLearningMiniTorch
