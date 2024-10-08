# import all packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import pickle


# analyse dataset
def analyse_dataset(y, folderPath):
    label_counts = y.value_counts()

    labels = label_counts.index.tolist()
    counts = label_counts.tolist()

    plt.figure(figsize=(6.4, 4.8))
    plt.bar(labels, counts)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Class Distribution in Dataset')
    plt.savefig(folderPath+'/class_dist.png')

# rf + knn function
def RF_KNN(dataset, label, folderPath):
    X = dataset.drop(label, axis=1)  # Features
    y = dataset[label]  # Labels

    analyse_dataset(y, folderPath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Standardize the data (important for KNN)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_importances = rf_model.feature_importances_
    important_features_indices = np.argsort(feature_importances)[::-1]

    top_features = 20

    # Select top-k features
    X_train_selected = X_train_scaled[:, important_features_indices[:top_features]]
    X_test_selected = X_test_scaled[:, important_features_indices[:top_features]]

    # Create a DataFrame to map features to their importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,  # Use column names from the original dataset
        'Importance': feature_importances
    })

    # Plot the selected features with their importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'].head(top_features), feature_importance_df['Importance'].head(top_features), color='#0e152f')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.title(f"Top {top_features} Features from Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.savefig(folderPath+'/features.png')

    accuracies = []
    knn_best_model = None
    maximum = -1

    for i in range(1,11):
        knn_model = KNeighborsClassifier(n_neighbors=i)
        knn_model.fit(X_train_selected, y_train)

        y_pred = knn_model.predict(X_test_selected)

        accuracy = accuracy_score(y_test, y_pred)

        accuracies.append(accuracy)

        # storing the best model
        if accuracy > maximum:
            knn_best_model = knn_model
            maximum = accuracy

    y_pred = knn_best_model.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(accuracies)
    print(accuracy)
    print(precision)
    print(recall)
    print(f1)

    with open(folderPath+"/rf_knn.pkl", 'wb') as file:
        pickle.dump(knn_best_model, file)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix for Random Forest + KNN')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(folderPath+'/cfm.png')

    plt.figure(figsize=(10, 6))
    plt.bar(range(1,11),accuracies, color ='#0e152f')
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Accuracies")
    plt.title("Accuracies for different KNN Neighbour values (1-10)")
    plt.savefig(folderPath+'/knn_neighbors.png')

    

# rf + svm function
def RF_SVM(dataset, label, folderPath):
    X = dataset.drop(label, axis=1)  # Features
    y = dataset[label]  # Labels

    analyse_dataset(y, folderPath)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)

    # Standardize the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    feature_importances = rf_model.feature_importances_
    important_features_indices = np.argsort(feature_importances)[::-1]

    top_features = 20

    # Select top-k features
    X_train_selected = X_train_scaled[:, important_features_indices[:top_features]]
    X_test_selected = X_test_scaled[:, important_features_indices[:top_features]]

    # Create a DataFrame to map features to their importance scores
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,  # Use column names from the original dataset
        'Importance': feature_importances
    })

    # Plot the selected features with their importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'].head(top_features), feature_importance_df['Importance'].head(top_features), color='#0e152f')
    plt.gca().invert_yaxis()  # Invert y-axis to display the most important feature at the top
    plt.title(f"Top {top_features} Features from Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.savefig(folderPath+'/features.png')

    svm_model = SVC()  # Enable probability for ROC-AUC
    svm_model.fit(X_train_selected, y_train)
    y_pred = svm_model.predict(X_test_selected)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(accuracy)
    print(precision)
    print(recall)
    print(f1)

    with open(folderPath+"/rf_svm.pkl", 'wb') as file:
        pickle.dump(svm_model, file)

    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title('Confusion Matrix for Random Forest + KNN')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(folderPath+'/cfm.png')



if __name__ == "__main__":
    # os.path.join(app.instance_path, 'datasets', filename)
    data = pd.read_csv('feature_vectors_syscallsbinders_frequency_5_Cat.csv')
    RF_KNN(data, 'Class', 'model1')
    RF_SVM(data, 'Class', 'model2')