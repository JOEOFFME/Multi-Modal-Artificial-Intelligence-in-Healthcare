# Multi-Modal-Artificial-Intelligence-in-Healthcare

A Computer Vision and Data-Driven Platform for Breast Cancer and Diabetes Diagnosis.

[![Python][python-shield]](#)
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

[python-shield]: https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white
[license-shield]: https://img.shields.io/github/license/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare?style=flat
[license-url]: https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/LinkedIn-Contact-blue?style=flat&logo=linkedin
[linkedin-url]: https://www.linkedin.com/in/joel-offome/ ---

## 🚀 About The Project

This repository explores the application of artificial intelligence in healthcare, focusing on two critical areas: **breast cancer** and **diabetes**. It leverages a multi-modal approach by analyzing both imaging data (Computer Vision) and structured tabular data (Data-Driven) to build robust diagnostic and exploratory platforms.

The project is divided into three main components:
1.  **Computer Vision for Mammography:** Using Convolutional Neural Networks (CNNs) to detect breast cancer from mammogram images.
2.  **Tabular Data Analysis for Breast Cancer:** Applying classical machine learning models to the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.
3.  **Tabular Data Analysis for Diabetes:** Exploring and preprocessing the Pima Indians Diabetes Database (PIMA).

---

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Projects & Methodology](#-projects--methodology)
  - [1. Mammography (Computer Vision)](#1-mammography-computer-vision)
  - [2. WDBC (Tabular ML)](#2-wdbc-tabular-ml)
  - [3. PIMA (Tabular EDA)](#3-pima-tabular-eda)
- [🔧 Technology Stack](#-technology-stack)
- [🏁 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [📄 License](#-license)
- [📧 Contact](#-contact)

---

## 🔬 Projects & Methodology

This repository contains three core projects, each detailed in its own Jupyter Notebook.

### 1. Mammography (Computer Vision)
- **Files:** `Mammography_processing.ipynb`, `Mammography_CNN.ipynb`
- **Goal:** To build and train a Convolutional Neural Network (CNN) to classify breast cancer from mammography images.
- **`Mammography_processing.ipynb`:** This notebook covers all necessary preprocessing steps for the image data, such as resizing, normalization, artifact removal, and data augmentation, to prepare a clean and robust dataset for the model.
- **`Mammography_CNN.ipynb`:** This notebook implements the CNN model architecture (using PyTorch). It details the model building, training, using transfer learning, and evaluation process, concluding with performance metrics like accuracy, precision, and recall.

### 2. WDBC (Tabular ML)
- **Files:** `WDBC FULL PROCESSING 30.ipynb`, `WDBC_Processed.csv`
- **Goal:** To explore the Wisconsin Diagnostic Breast Cancer (WDBC) dataset and build a machine learning model to classify tumors as malignant or benign.
- **`WDBC FULL PROCESSING 30.ipynb`:** This notebook provides a complete workflow, including:
    - Exploratory Data Analysis (EDA)
    - Data cleaning and preprocessing
    - Feature engineering and selection
    - Training and comparing multiple machine learning models using Stacking approches
    - Evaluating the final model's performance.
- **`WDBC_Processed.csv`:** A cleaned version of the dataset, ready for modeling.

### 3. PIMA (Tabular EDA)
- **Files:** `PIMA Preprocessing and exploring .ipynb`, `PIMA_Processed.csv`
- **Goal:** To perform in-depth exploratory data analysis (EDA) and preprocessing on the Pima Indians Diabetes Database.
- **`PIMA Preprocessing and exploring .ipynb`:** This notebook focuses on understanding the dataset, visualizing relationships between features, handling missing values, and preparing the data for future machine learning tasks. It serves as the foundation for building a diabetes prediction model.
- **`PIMA_Processed.csv`:** A cleaned version of the PIMA dataset.

---

## 🔧 Technology Stack

This project is built using the following technologies:

- **Python:** The core programming language.
- **Jupyter Notebook:** For interactive development and analysis.
- **Pandas:** For data manipulation and analysis of tabular data.
- **NumPy:** For numerical operations.
- **Scikit-learn:** For classical machine learning models (SVM, Random Forest, etc.) and metrics.
- **TensorFlow & Keras:** For building and training the Convolutional Neural Network.
- **Matplotlib & Seaborn:** For data visualization.
- **OpenCV (cv2):** (Likely used in `Mammography_processing.ipynb`) For image processing tasks.

---

## 🏁 Getting Started

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
    pip install pandas numpy scikit-learn tensorflow matplotlib seaborn jupyter opencv-python
    ```

4.  **Launch Jupyter Notebook:**
    ```sh
    jupyter notebook
    ```
    You can now open and run the `.ipynb` files to see the analysis and models in action.

---

## 📄 License

Distributed under the MIT License. See `LICENSE.txt` for more information. (You will need to add a file named `LICENSE.txt` or `LICENSE` to your repo with the MIT License text).

---

## 📧 Contact

JOEOFFME - [Your LinkedIn](https://www.linkedin.com/in/joel-offome/) Project Link: [https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare](https://github.com/JOEOFFME/Multi-Modal-Artificial-Intelligence-in-Healthcare)
