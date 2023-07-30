## PhishNet

### Description:
This project applies a Deep Learning approach to detect phishing websites. A PyTorch-based Feed-Forward Neural Network is trained to classify websites based on various features such as favLabel, percentRequestUrls, URLLength, domainLabel, and dotsInDomain.

The dataset used in this project comprises various website characteristics and a label indicating whether the website is a phishing website or not.

The project includes data loading, preprocessing, model training, and testing with the implementation of early stopping for optimal performance.

### Main Components:
1. Data Loading and Preprocessing: Data is loaded from a CSV file and preprocessed for the model. This includes a train-test split and normalization of the features.

2. Model Training: The project employs a three-layer feed-forward neural network. The training process uses the Adam optimizer and CrossEntropyLoss.

3. Validation and Early Stopping: The model's performance is validated on a separate dataset during the training process. If the validation loss does not improve for a certain number of epochs, the training process is stopped to prevent overfitting.

4. Model Testing: The model is tested on a separate test dataset to evaluate its performance.

### Potential Enhancements:
- Implementation of other neural network architectures
- Feature engineering or selection for improving model performance
- Hyperparameter tuning
- Cross-validation for model performance estimation

### Usage:
Please refer to the phishing_detection.py script for the main implementation.

Note: Please modify the path for data loading and model saving based on your directory structure. The project assumes that data is in a file named 'final_dataset.csv' and the best model will be saved as 'best_model.pt'.

### Acknowledgments:
This is a baseline model for phishing website detection. For further improvement, additional techniques in feature engineering, advanced modeling, and validation can be applied.
