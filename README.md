# Comparative Analysis of Shallow versus Deep Learning Architectures for Network Intrusion Detection Systems

**Authors:** Rijul Sharma, Saurabh Verma, Shiva Chahar  
**Roll No:** 2210992152, 2210992271, 2210992301
**Institution:** Chitkara University, Punjab, India  
**Type:** Research Paper
---

## 📌 Overview
This repository contains the source code and experimental results for our research paper evaluating the efficacy of Machine Learning (Random Forest) versus Deep Learning (1D-CNN) for Network Intrusion Detection Systems (NIDS). 

Traditional signature-based IDS are increasingly ineffective against zero-day exploits. While Deep Learning is highly popular, our research challenges the assumption that complex neural networks are the optimal choice for **tabular network data**. We utilized the benchmark **NSL-KDD dataset** to evaluate both paradigms based on accuracy, precision, recall, and computational training latency.

## 📊 Key Findings
Our empirical results demonstrate that shallow ensemble methods natively handle tabular network data better than baseline deep learning models. 
* **Random Forest** achieved a slightly higher accuracy (**77.04%**) in a fraction of the time (**7.33 seconds**).
* **1D-CNN** achieved **76.38%** accuracy but required significantly higher computational overhead (**25.07 seconds** on a GPU).
* Both models struggled with minority-class attack recall, highlighting the need for hybrid architectures and dataset balancing (SMOTE) in future work.

### Performance Metrics
| Metric | Random Forest (RF) | 1D Convolutional Neural Network (CNN) |
| :--- | :--- | :--- |
| **Training Time** | **7.33 seconds** | 25.07 seconds |
| **Overall Accuracy** | **77.04%** | 76.38% |
| **Precision (Attack)** | **0.97** | 0.97 |
| **Recall (Attack)** | **0.62** | 0.60 |
| **F1-Score (Attack)** | **0.75** | 0.74 |

---

## ⚙️ Dataset
We used the **NSL-KDD** dataset, which resolves the massive redundancy issues present in the legacy KDD'99 dataset, ensuring unbiased classifier evaluation.
* The script automatically downloads the `KDDTrain+.txt` and `KDDTest+.txt` files directly from the source.
* **Preprocessing applied:** Categorical Label Encoding and Min-Max Normalization.

---

## 🚀 How to Run the Code

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries using:
```bash
pip install -r requirements.txt
