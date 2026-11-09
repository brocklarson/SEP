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
        * Three encoder options to compare  
            * 1-D Convolutional Encoder  
            * GRU Encoder  
            * MLP using catch22 extracted features
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
            * learning rate: 10 ** rng.uniform(-5, -3)  
            * weight decay: 10 ** rng.uniform(-2.5, -1.0)  
            * margin: rng.uniform(0.2, 0.6)  
            * dropout: rng.choice([0.2, 0.3, 0.4, 0.5])  
            * hidden channels: rng.choice([32, 64, 128])  
            * embedding dimensions: rng.choice([16, 32])  
            * hard anchor fraction: rng.uniform(0.0, 0.8)  
            * hard negatives pool: 10 ** rng.uniform(2.0, 3.5)  
            * include hard negatives: rng.choice([True, False])  
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
    * Results  
        * __A random search was conducted to find the best model configuration. The search comprised 15 seeds, repeated across 3 encoders and 2 random state seeds. Each encoder and state was also tested on an untrained model.__  
        * Logistic Regression  
            * TSS: 0.667 ± 0.021  
        * Centroid  
            * TSS: 0.637 ± 0.026  
        * kNN  
            * TSS: 0.637 ± 0.025  
        * SVM  
            * TSS: 0.624 ± 0.013   
        * XGBoost  
            * TSS: 0.619 ± 0.046  

6. InfoNCE
    * Configuration
        * Two loss types to compare
            * SimCLR 
            * SupCon
        * Augmentation 
            * Gaussian Jitter
            * Random per-channel scaling
        * GRU Encoder
        * Batch Sampling
            * Ensures a fixed fraction of each batch are SEP examples
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
            * learning rate: 10 ** rng.uniform(-5, -3)  
            * weight decay: 10 ** rng.uniform(-5, -2)  
            * temperature: rng.choice([0.1, 0.2, 0.5, 1.0])  
            * hidden channels: rng.choice([16, 32, 64])  
            * embedding dimensions: rng.choice([16, 32])
            * batch size: rng.choice([128, 256])  
            * SEP fraction: rng.uniform(0.1, 0.45)  
            * noise std: rng.choice([0.05, 0.1, 0.2])
            * scale std: rng.choice([0.05, 0.1, 0.2])  
        * Classifier specific tuning  
            * Logistic Regression   
                * C: 10 ** rng.uniform(-3, 0.5)  
            * kNN    
                * k: rng.choice([20, 40, 60, 80, 100])  
            * SVM  
                * C: 10 ** rng.uniform(-3, 0.5)  
                * gamma: 10 ** rng.choice(["scale", 0.03, 0.1, 0.3, 1.0])  
            * XGBoost  
                * max depth: rng.choice([2, 3, 4, 6])  
                * min child weight: rng.choice([10, 20, 40, 800])  
                * n estimators: rng.choice([50, 100, 200])   
                * learning rate: 10 ** rng.uniform(-2.5, -1)  
                * colsample bytree: rng.choice([0.3, 0.5])    
    * Results  
        * __A random search was conducted to find the best model configuration. The search comprised 15 seeds, repeated across 2 random state seeds. Each encoder and state was also tested on an untrained model.__  
        * Logistic Regression  
            * TSS: 0.669 ± 0.016  
        * Centroid  
            * TSS: 0.669 ± 0.019  
        * kNN  
            * TSS: 0.544 ± 0.072  
        * SVM  
            * TSS: 0.638 ± 0.036   
        * XGBoost  
            * TSS: 0.516 ± 0.071  


To Try: 
* RUS  
* K cross-fold training set  
