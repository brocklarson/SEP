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

4. Test baseline classifiers  
    * Logistic Regression 
        * Tune the C, class_weight, and threshold parameters on validation set, then test on test set
        * TSS: 0.4297 ± 0.0000  
    * XGBoost  
        * Tune the learn_rate, scale_pos_weight, max_depth, min_child_weight, and colsample_bytree, then test on test set
        * TSS: 0.6545 ± 0.0080  
    * GRU
        * Tune the units, dropout, layers, learning rate, and batch size, then test on test set
        * TSS: 0.5951 ± 0.0323  
    * SVM
        * Extract catch22 features and tune the class weight, gamma, and kernel, then test on test set
        * TSS: 0.4876 ± 0.0000 
    * MLP
        * Tune the dropout, weight decay, learning rate, and batch size, then test on test set
        * TSS: 0.5195 ± 0.0134 
    * LSTM
        * Tune the dropout, layers, learning rate, and batch size, then test on test set
        * TSS: 0.6266 ± 0.0151 
    * RNN
        * Tune the dropout, layers, learning rate, and batch size, then test on test set
        * TSS: 0.6130 ± 0.0318 
    * CNN
        * Tune the kernel, dropout, layers, learning rate, and batch size, then test on test set
        * TSS: 0.4227 ± 0.0751 

5. Triplet Loss  
    * Configuration   
        * Triplet loss using Euclidean distance  
        * Three encoder options to compare  
            * 1-D Convolutional Encoder  
            * GRU Encoder  
            * MLP Encoder using catch22 extracted features
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
        * How much is contrastive learning adding? 
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
        * Epochs = (steps × batch size) / training set size => (256 * 1000) / 12,455 ≈ 20.6 epochs
            * Approximate because training is done for an amount of gradient steps
    * Results  
        * __A random search was conducted to find the best model configuration. The search comprised 15 seeds, repeated across 3 encoders and 2 random state seeds. Each encoder and state was also tested on an untrained model.__  
        * Logistic Regression  
            * TSS: 0.691 ± 0.034      
        * Centroid  
            * TSS: 0.637 ± 0.026  
        * kNN  
            * TSS: 0.637 ± 0.025  
        * SVM  
            * TSS: 0.624 ± 0.013   
        * XGBoost  
            * TSS: 0.619 ± 0.046 

To Try: 
* Different Triplet Mining 
* RUS  
* K cross-fold training set  
