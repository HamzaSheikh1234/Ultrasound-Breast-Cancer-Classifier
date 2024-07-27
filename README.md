# Ultrasound Breast Cancer Classifier

This project focuses on developing a machine learning model to classify breast cancer using ultrasound images. By leveraging convolutional neural networks (CNNs), the classifier aims to aid in the early detection and diagnosis of breast cancer.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

Breast cancer is one of the most common cancers affecting women worldwide. Early detection through reliable diagnostic methods can significantly improve treatment outcomes. This project utilizes deep learning techniques to classify ultrasound images of breast tumors as benign or malignant.

## Features

- **Data Preprocessing**: Includes image normalization, augmentation, and resizing.
- **Model Training**: Implements a CNN for binary classification.
- **Evaluation Metrics**: Accuracy, precision, recall, and F1-score.
- **Visualization**: Plots for training history, confusion matrix, and sample predictions.

## Dataset

The dataset used for this project contains ultrasound images of breast tumors, labeled as benign or malignant. The images are preprocessed and split into training, validation, and test sets.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HamzaSheikh1234/Ultrasound-Breast-Cancer-Classifier.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Ultrasound-Breast-Cancer-Classifier
    ```
3. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Data Preparation**: Ensure the dataset is organized in the specified format and update the dataset path in the configuration file.

2. **Training the Model**:
    ```bash
    python train.py
    ```

3. **Evaluating the Model**:
    ```bash
    python evaluate.py
    ```

4. **Visualizing Results**:
    ```bash
    python visualize.py
    ```

## Model Architecture

The model is a convolutional neural network (CNN) designed for binary classification of ultrasound images. It consists of multiple convolutional layers followed by pooling layers, and fully connected layers with dropout for regularization.

## Results

The model's performance is evaluated using various metrics such as accuracy, precision, recall, and F1-score. Visualizations include training/validation loss and accuracy curves, confusion matrix, and sample prediction outputs.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of the changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any inquiries or issues, please open an issue in the repository or contact the project maintainer:

- **Hamza Sheikh** - [GitHub Profile](https://github.com/HamzaSheikh1234)
