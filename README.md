# Overview   

### Notebooks
* preprocessing.ipynb
    * Load data from csv files
    * Split into chronological train/val/test sets
    * Apply a hybrid normalization
    * Save in processed data directory
* baseline.ipynb
    * Apply various classifiers to normalized data
* triplet_loss.ipynb
    * Apply triplet loss models to normalized data
* infonce.ipynb
    * Apply InfoNCE models to normalized data

### Pipeline  
1. Compile 10 raw per-feature CSVs into a usable tensor  
    * Read the last 288 columns from each csv and check for NaN or inf values  
    * Convert labels into a binary classification of SEP and NSEP  
    * Stack in resulting X tensor of shape (N, 288, 10) and a y tensor of shape (N,) 

2. Split dataset chronologically (70/10/20) while maintaining class imbalance across sets  

3. Normalize dataset per feature (Two separate approaches in codebase until I settle on one over the other)  
    * Hybrid normalization where the applied normalization is based on the statistical behavior of the channel  
        * $F$ : Log + Z-score  
        * $N_p$ : Log + Z-score  
        * $P_4$ : Log + Z-score
        * $P_5$ : Log + Z-score
        * $P_6$ : Log + Z-score
        * $T_p$ : Log + Z-score
        * $V$ : Z-score
        * $V_x$ : Z-score
        * $Xl$ : Log + Z-score
        * $Xs$ : Log + Z-score

4. Test baseline classifiers (using hybrid normalization)  
    * Logistic Regression 
        * Tune the C, class_weight, and threshold parameters on validation set, then test on test set
        * Tune the classification threshold based on the ROC curve to: 0.4818
        * TSS: 0.42946
    * XGBoost  
        * Tune the learn_rate, scale_pos_weight, max_depth, min_child_weight, and colsample_bytree, then test on test set
        * Tune the classification threshold based on the ROC curve to: 0.00720
        * TSS: 0.64839
    * GRU
        * Tune the units, dropout, layers, learning rate, and batch size, then test on test set
        * Tune the classification threshold based on the ROC curve to: 0.6647
        * TSS: 0.60801 
    * SVM
        * Extract catch22 features and tune the class weight, gamma, and kernel, then test on test set
        * TSS: 0.48763

5. Triplet Loss 
    * Configuration
        * Triplet loss using Euclidean distance
        * Two encoder options to compare
            * 1-D Convolutional Encoder
            * GRU Encoder
        * Anchor selection using hard and semi-hard NSEPs
            * Refresh hard negative pool every 250 steps as embedding space changes
        * Tune and evaluate various downstream classifiers
            * Logistic Regression
            * Centroid
            * kNN
            * SVM
            * XGBoost
    * Primary Challenge
        * Overfitting   
    * Experiments
        * Run each classifier without training to ensure triplet loss is adding value
        * Random search grid tuning 
            * learning rate: 10 ** rng.uniform(-4, -2.5)
            * weight decay: 10 ** rng.uniform(-3, -1.3)
            * margin: rng.uniform(0.1, 0.6)
            * dropout: rng.choice([0.0, 0.1, 0.2])
            * hidden channels: rng.choice([16, 32, 64])
            * embedding dimensions: rng.choice([16, 32, 64])
            * hard anchor fraction: rng.uniform(0.0, 0.8)
            * hard negatives pool: 10 ** rng.uniform(2.3, 3.5)
            * include hard negatives: rng.choice([False, True])
        * Classifier specific tuning
            * kNN  
                * k: rng.choice([5, 10, 20, 40, 80])
            * SVM
                * kernel: rng.choice(["linear", "rbf"])
                * C: 10 ** rng.uniform(-2, 1)
                * gamma: 10 ** rng.uniform(-3, -1)
            * XGBoost
                * max depth: rng.choice([2, 3, 4, 6])
                * min child weight: rng.choice([1, 5, 10, 20])
                * n estimators: rng.choice([100, 200, 400])
                * learning rate: 10 ** rng.uniform(-2, -0.7)
                * colsample bytree: rng.choice([0.3, 0.5, 0.8, 1.0])




To Try: 
* MLP encoder with catch 22 feature extraction
* RUS
* K cross-fold training set
