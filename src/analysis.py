import os
import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy.stats import chi2

#Import the classifiers we defined in classification.py
from classification import get_classifiers

# --- Configuration & Setup ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "augmented_images")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

CONF_MATRIX_DIR = os.path.join(RESULTS_DIR, "confusion_matrices")
ROBUSTNESS_DIR = os.path.join(RESULTS_DIR, "robustness_plots")

#Ensure output directories exist
os.makedirs(CONF_MATRIX_DIR, exist_ok=True)
os.makedirs(ROBUSTNESS_DIR, exist_ok=True)

CLASSES = ["google_docs_pdfs", "python_pdfs", "word_pdfs"]
CONDITIONS =["original", "gaussian", "jpeg", "dpidown", "crop", "bitdepth"]

def get_label_from_filename(filename):
#Extracts the integer label based on the class prefix.
    for i, cls in enumerate(CLASSES):
        if filename.startswith(cls):
            return i
    return -1

def load_image_data(condition, allowed_bases=None, img_size=(128, 128)):
    """
    Loads images for a specific condition.
    If allowed_bases is provided, only loads images whose original base name is in the list
    (used to enforce strict train/test separation across all augmentations).
    """
    X, y = [],[]
    search_pattern = os.path.join(DATA_DIR, f"*_{condition}.png")
    #Also check .jpg just in case
    search_pattern_jpg = os.path.join(DATA_DIR, f"*_{condition}.jpg")
    
    files = glob.glob(search_pattern) + glob.glob(search_pattern_jpg)
    
    for file_path in files:
        filename = os.path.basename(file_path)
        
        #Extract base name (e.g., "word_pdfs_doc1" from "word_pdfs_doc1_original.png")
        base_name = filename.rsplit(f"_{condition}", 1)[0]
        
        if allowed_bases is not None and base_name not in allowed_bases:
            continue
            
        label = get_label_from_filename(filename)
        if label == -1:
            continue
            
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            #Resize to a fixed size to prevent memory explosion when flattening
            img_resized = cv2.resize(img, img_size)
            X.append(img_resized.flatten())
            y.append(label)
            
    return np.array(X), np.array(y)

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    Performs McNemar's test to compare two classifiers' predictions.
    Returns the p-value.
    """
    # b: clf1 correct, clf2 wrong
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    # c: clf1 wrong, clf2 correct
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    if b + c == 0:
        return 1.0
        
    #Calculate test statistic with continuity correction
    stat = ((np.abs(b - c) - 1)**2) / (b + c)
    p_value = chi2.sf(stat, 1) #Survival function (1 - CDF) for chi-square with 1 df
    return p_value

def main():
    print("--- Starting Comprehensive Analysis Pipeline ---")
    
    #1. Identify all unique base images to create a strict Train/Test split
    #This ensures the model never sees an augmented version of an image it trained on.
    all_original_files = glob.glob(os.path.join(DATA_DIR, "*_original.*"))
    all_bases =[os.path.basename(f).rsplit("_original", 1)[0] for f in all_original_files]
    
    if not all_bases:
        print(f"Error: No images found in {DATA_DIR}")
        return

    train_bases, test_bases = train_test_split(all_bases, test_size=0.2, random_state=42)
    print(f"Dataset split: {len(train_bases)} training images, {len(test_bases)} testing images.")

    #2. Load Training Data (Original ONLY, as per Task 3 requirements)
    X_train, y_train = load_image_data("original", allowed_bases=train_bases)
    
    #Scale the data (Standard ML practice for SVM/SGD/MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    #3. Initialize and Train Classifiers
    classifiers = get_classifiers()
    
    print("\n--- Training Classifiers ---")
    for name, clf in classifiers.items():
        print(f"Training {name}...")
        clf.fit(X_train_scaled, y_train)

    #4. Evaluate on all conditions (Domain Shift / Robustness Testing)
    print("\n--- Evaluating Robustness ---")
    results =[]
    predictions_dict = {name: {} for name in classifiers.keys()} #Store preds for statistical testing
    y_true_dict = {}

    for condition in CONDITIONS:
        print(f"Evaluating condition: {condition}")
        X_test, y_test = load_image_data(condition, allowed_bases=test_bases)
        
        if len(X_test) == 0:
            print(f"  -> Skipping {condition}, no data found.")
            continue
            
        X_test_scaled = scaler.transform(X_test)
        y_true_dict[condition] = y_test
        
        for name, clf in classifiers.items():
            y_pred = clf.predict(X_test_scaled)
            predictions_dict[name][condition] = y_pred
            
            #Metrics
            acc = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)
            
            results.append({
                "Classifier": name,
                "Condition": condition,
                "Accuracy": acc,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1
            })
            
            #Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
            plt.title(f'Confusion Matrix: {name} ({condition})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(CONF_MATRIX_DIR, f"{name}_{condition}_cm.png"))
            plt.close()

    #Save metrics to CSV
    df_results = pd.DataFrame(results)
    csv_path = os.path.join(RESULTS_DIR, "performance_metrics.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}")

    #5. Generate Robustness Curves (Task 3)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_results, x='Condition', y='Accuracy', hue='Classifier', marker='o')
    plt.title('Classifier Robustness Across Augmentations')
    plt.ylabel('Accuracy')
    plt.xlabel('Augmentation Condition')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Classifier')
    plt.tight_layout()
    robustness_plot_path = os.path.join(ROBUSTNESS_DIR, "robustness_curves.png")
    plt.savefig(robustness_plot_path)
    plt.close()
    print(f"Robustness curves saved to {robustness_plot_path}")

    #6. Statistical Significance Testing (Task 5)
    print("\n--- Statistical Significance Testing (McNemar's Test) ---")
    print("Comparing new classifiers against baseline SVM on the ORIGINAL test set:")
    
    if "original" in y_true_dict:
        y_true_orig = y_true_dict["original"]
        preds_svm = predictions_dict["SVM"]["original"]
        
        for new_clf in ["Random_Forest", "MLP"]:
            if new_clf in predictions_dict:
                preds_new = predictions_dict[new_clf]["original"]
                p_val = mcnemar_test(y_true_orig, preds_svm, preds_new)
                
                significance = "Significant difference" if p_val < 0.05 else "No significant difference"
                print(f"  SVM vs {new_clf}: p-value = {p_val:.4f} ({significance})")

    print("\nPipeline Complete! Check the 'results' folder for your deliverables.")

if __name__ == "__main__":
    main()