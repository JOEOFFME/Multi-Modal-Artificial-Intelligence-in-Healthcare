# Multi-Modal-Artificial-Intelligence-in-Healthcare

A Computer Vision and Data-Driven Platform for Breast Cancer and Diabetes Diagnosis.

[![Python][python-shield]](#)
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

[python-shield]: https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white
[license-shield]: https://img.shields.io/github/license/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare?style=flat
[license-url]: https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-Contact-blue?style=flat&logo=linkedin
[linkedin-url]: https://www.linkedin.com/in/youssef-dihaji-8458b0310/overlay/about-this-profile/

##  About The Project

This repository explores the application of artificial intelligence in healthcare, focusing on two critical areas: **breast cancer** and **diabetes**. It leverages a multi-modal approach by analyzing both imaging data (Computer Vision) and structured tabular data (Data-Driven) to build robust diagnostic and exploratory platforms.

The project is divided into three main components:
1.  **Computer Vision for Mammography:** Using **PyTorch** and **Transfer Learning** to detect breast cancer.
2.  **Tabular Data Analysis for Breast Cancer:** Applying classical machine learning models to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
3.  **Tabular Data Analysis for Diabetes:** Building a predictive model using the Pima Indians Diabetes Database (PIMA).

---

##  Table of Contents

- [About The Project](#-about-the-project)
- [Projects & Methodology](#-projects--methodology)
  - [1. Mammography (Computer Vision)](#1-mammography-computer-vision)
  - [2. WDBC Breast Cancer (Tabular ML)](#2-wdbc-breast-cancer-tabular-ml)
  - [3. PIMA Diabetes Prediction (Tabular ML)](#3-pima-diabetes-prediction-tabular-ml)
- [ Technology Stack](#-technology-stack)
- [ Getting Started](#-getting-started)
- [ License](#-license)
- [ Contact](#-contact)

---

## Projects & Methodology

This repository contains three core projects, each detailed in its own Jupyter Notebook.

### 1. Mammography (Computer Vision)
- **Files:** `Mammography_processing.ipynb`, `Mammography_CNN.ipynb`
- **Goal:** To build and train a model to classify breast cancer from mammogram images using **PyTorch** and **Transfer Learning**.
- **`Mammography_processing.ipynb`:** This notebook covers all necessary preprocessing steps for the image data, such as resizing, normalization, artifact removal, and data augmentation, to prepare a clean and robust dataset for the model.
- **`Mammography_CNN.ipynb`:** This notebook implements the model using **PyTorch**. It leverages **Transfer Learning** (e.g., using a pre-trained model like ResNet or VGG) to achieve high accuracy. It details the model fine-tuning, training, and evaluation process, concluding with performance metrics.

### 2. WDBC Breast Cancer (Tabular ML)
- **Files:** `WDBC FULL PROCESSING 30.ipynb`, `WDBC_model_30.pkl`, `WDBC_Processed.csv`
- **Goal:** To explore the Wisconsin Diagnostic Breast Cancer (WDBC) dataset and build a machine learning model to classify tumors as malignant or benign.
- **`WDBC FULL PROCESSING 30.ipynb`:** This notebook provides a complete workflow, including EDA, preprocessing, feature selection, and training/comparing multiple ML models using staking approches (e.g., Logistic Regression, SVM, Random Forest).
- **`WDBC_model_30.pkl`:** This is the saved, trained machine learning model for classifying tumors.
- **`WDBC_Processed.csv`:** A cleaned version of the dataset, ready for modeling.

### 3. PIMA Diabetes Prediction (Tabular ML)
- **Files:** `PIMA Preprocessing and exploring .ipynb`, `PIMA_Modeling (3) (1).ipynb`, `Diabetes_scaler.pkl`, `Diabetes_model.pkl`
- **Goal:** To perform EDA and build a machine learning model to predict the onset of diabetes based on the Pima Indians Diabetes Database.
- **`PIMA Preprocessing and exploring .ipynb`:** This notebook focuses on in-depth EDA, visualizing relationships between features, and handling missing values.
- **`PIMA_Modeling (3) (1).ipynb`:** This notebook covers the model building, training, and evaluation for diabetes prediction.
- **`Diabetes_model.pkl` & `Diabetes_scaler.pkl`:** These are the final saved predictive model and the scikit-learn scaler, allowing for future predictions on new data.
- **`PIMA_Processed.csv`:** A cleaned version of the PIMA dataset.

---

##  Technology Stack

This project is built using the following technologies:

- **Python:** The core programming language.
- **Jupyter Notebook:** For interactive development and analysis.
- **Pandas:** For data manipulation and analysis of tabular data.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For classical machine learning models (SVM, Random Forest, etc.), data scaling, and metrics.
- **PyTorch:** For building and training the Convolutional Neural Network using transfer learning.
- **Matplotlib & Seaborn:** For data visualization.
- **OpenCV (cv2) & Pillow (PIL):** For image processing and loading tasks.

---

##  Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

Ensure you have Python 3.7+ and pip installed.

### Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare.git](https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare.git)
    ```

2.  **Navigate to the project directory:**
    ```sh
    cd Multi-Modal-Artificial-Intelligence-in-Healthcare
    ```

3.  **Install the required packages:**
    (It's highly recommended to use a virtual environment)
    ```sh
    pip install pandas numpy scikit-learn torch torchvision matplotlib seaborn jupyter opencv-python pillow
    ```

4.  **Launch Jupyter Notebook:**
    ```sh
    jupyter notebook
    ```
    You can now open and run the `.ipynb` files to see the analysis and models in action.

---

##  License

Distributed under the MIT License. See `LICENSE` for more information.

---

## S Contact

JOEOFFME - [LinkedIn](https://www.linkedin.com/in/joel-offome/) Project Link: [https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare](https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare)
