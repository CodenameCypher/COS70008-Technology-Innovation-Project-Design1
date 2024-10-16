# MLassify - COS70008 Technology Innovation Project (Design 1)

This project is a web-based system developed for malware detection and classification using a hybrid machine learning model. It provides different functionalities for admin and normal users, focusing on analyzing behavioral patterns to detect malicious activities.

## Project Overview

### System Design Specifications:

- **Web Framework**: Flask (Python)
- **Database System**: SQLite, Flask SQLAlchemy (ORM)
- **User Interface**: Jinja2 Template Engine, HTML, Bootstrap
- **ML Packages**: `numpy`, `matplotlib`, `pandas`, `scikit-learn`, `seaborn`, `pickle`
- **Design Framework**: MVC (Model View Controller)
- **Version Control**: GitHub

### Machine Learning Model Workflow:

The Hybrid Machine Learning Model follows this training flow:

1. **Dataset**: Input dataset containing application behavior or system data.
2. **Random Forest**: Used for feature selection, identifying the most important features.
3. **Scale the Reduced Dataset**: Apply a scaler to normalize the dataset.
4. **KNN/SVM Training**: Train KNN or SVM model with the selected features.
5. **Final Model**: The final model is saved for future predictions.

## Functional Requirements

### Admin Users:

1. **Authentication & Logout**: Secure login/logout to access the system.
2. **Upload Datasets**: Ability to upload datasets for training the machine learning models.
3. **Algorithm Selection**: Select the algorithm (KNN or SVM) and dataset for training.
4. **Visualize Analysis**: View analysis results through various visualizations (e.g., feature importance, prediction results).
5. **User Management**: Manage user accounts and roles in the system.
6. **Profile Management**: Update and manage own profile.

### Normal Users:

1. **Authentication & Logout**: Secure login/logout to access the system.
2. **View Analysis**: Ability to view the analysis results and visualizations.
3. **Classification Input**: Input parameters (features) to classify outcomes using the saved trained models (KNN/SVM).
4. **Profile Management**: Update and manage own profile.

## Limitations

1. **Model Accuracy**: The hybrid model of Random Forest + SVM currently achieves around **67% accuracy**.
2. **Manual Input for Classification**: Users must manually input the behavior (feature column values) of the application for classification. There is no file upload feature for malware classification/prediction yet.

## Intended Future Work

1. **Model Improvement**: Improve the model accuracies by exploring different machine learning strategies (e.g., feature engineering, hyperparameter tuning).
2. **File Upload Feature**: Implement functionality that allows users to upload files for malware classification.
3. **Dynamic Programming**: Implement a dynamic programming approach to optimize system efficiency and reduce the computation time for large datasets.

## How to Run the Project

### Prerequisites

Ensure you have Python and the required packages installed. You can install the dependencies using:

bash
pip install -r requirements.txt

### Running the Web Application

1. Clone the repository:
    
```bash
git clone https://github.com/your-username/malware-detection.git
```
   
2. Navigate to the project directory:

3. Run the Flask application:
    
```bash
python app.py
```

4. Access the application by navigating to `http://127.0.0.1:5000/` in your web browser.

### Version Control

- The project is maintained on GitHub, and all changes are tracked using Git.

---

## Contributors

- **[Aditya Roy]** - Developer & Maintainer
