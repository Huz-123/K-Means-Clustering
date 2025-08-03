# K-Means Clustering Projects

This repository contains two Jupyter Notebook projects demonstrating K-Means clustering on different datasets:

1. **Income Data K-Means Clustering** (`K Means Clustering_income_data_project_1.ipynb`)
2. **Iris Data K-Means Clustering Exercise** (`K Means Clustering_income_data_project_2.ipynb`)

The overall workflow for both projects follows the steps illustrated in the included flowchart (Image 1).

---

## 🧑‍💻 General Workflow

The projects follow these key steps:

1. **Import Libraries**  
   Core Python libraries (pandas, numpy, matplotlib, seaborn, scikit-learn) are imported.

2. **Load Dataset**  
   - Project 1: Loads a CSV file of income data.
   - Project 2: Loads the Iris dataset from scikit-learn.

3. **Explore Data**  
   - Displays the dataset.
   - Checks columns and statistics.
   - Plots data to visually inspect groupings.

4. **Preprocess Data**  
   - May involve feature scaling, normalization, or dropping unnecessary columns.
   - Project 1 uses `MinMaxScaler` to scale age and income.
   - Project 2 drops unused features for simplicity.

5. **Apply K-Means Clustering**  
   - Uses `KMeans` from scikit-learn.
   - Data is fit and cluster labels are predicted.

6. **Choose Number of Clusters (k)**  
   - Uses the "Elbow Method":  
     - Fits KMeans models for a range of k values.
     - Plots the sum of squared errors (SSE) vs k to find the optimal number of clusters.

7. **Fit KMeans Model and Assign Labels**  
   - Best k is chosen and final labels assigned to each data point.

8. **Inspect Cluster Centers**  
   - Cluster centroids are printed and analyzed.

9. **Visualize Clusters**  
   - Scatter plots show clusters in feature space.
   - Cluster centers marked.

10. **Interpret Results**  
    - Discusses separability, cluster characteristics, and results.

---

## 🔎 Project Details

### 1. Income Data K-Means Clustering

- **Dataset:**  
  Artificial dataset with `Name`, `Age`, and `Income($)`.
- **Objective:**  
  Divide people into clusters based on age and income.
- **Preprocessing:**  
  Features are scaled using MinMaxScaler.
- **Clustering:**  
  KMeans applied, clusters visualized, and cluster centroids printed.
- **Elbow Plot:**  
  Used to determine optimal k.

### 2. Iris Data K-Means Clustering

- **Dataset:**  
  Classic Iris dataset; only `petal length` and `petal width` are used.
- **Objective:**  
  Cluster flowers by petal features.
- **Preprocessing:**  
  Unused features are dropped.
- **Clustering:**  
  KMeans applied, clusters visualized, and centroids inspected.
- **Elbow Plot:**  
  Used to select k.

---

## 🖼️ Flowchart Reference

The included **flowchart (Image 1)** summarizes the above pipeline:

```
START
  ↓
IMPORT_LIBRARIES
  ↓
LOAD_DATASET
  ↓
EXPLORE_DATA
  ↓
PREPROCESS_DATA
  ↓
APPLY_KMEANS_CLUSTERING
  ↓
CHOOSE_NUMBER_OF_CLUSTERS_K
  ↓
FIT_KMEANS_MODEL_AND_ASSIGN_LABELS
  ↓
INSPECT_CLUSTER_CENTERS
  ↓
VISUALIZE_CLUSTERS
  ↓
INTERPRET_RESULTS
  ↓
END
```

---

## 🏁 How to Run

1. Open any notebook in Jupyter or Google Colab.
2. Step through the code cells, following the workflow above.
3. Adjust dataset paths (for Project 1) as needed.

---

## 📚 Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

---

## ✨ Key Learnings

- How to use K-Means for unsupervised clustering.
- Importance of feature scaling.
- How to choose the number of clusters using the elbow method.
- Visualization and interpretation of clusters.

---

## 📂 Files

- `K Means Clustering_income_data_project_1.ipynb` - Income Clustering Project
- `K Means Clustering_income_data_project_2.ipynb` - Iris Clustering Exercise
- **Image 1** - Project workflow flowchart

---

## 🚀 Next Steps

- Try with other datasets or features.
- Experiment with different preprocessing steps.
- Compare clustering results with ground truth labels (where available).

---

**For more details, see each notebook and the flowchart (Image 1) for the structured workflow.**
