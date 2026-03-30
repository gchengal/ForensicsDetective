Project Setup Documentation (Task 1) 
This document outlines the steps taken to set up the working environment, fork the base repository, and verify the initial `ForensicsDetective` pipeline before implementing the advanced scaling and robustness features for Assignment 2. 
## 1. Repository Forking and Cloning 
1. Navigated to the original repository at `https://github.com/delveccj/ForensicsDetective`. 
2. Read the `STUDENTS_START_HERE.md` guide to understand the baseline PDF-to-image conversion approach. 
3. Created a personal fork of the repository into my own GitHub workspace. 
4. Cloned the forked repository to my local machine using: ```bash git clone https://github.com/gchengal/ForensicsDetective.git cd ForensicsDetective

## 2. Environment Configuration

To ensure a clean working environment and avoid dependency conflicts, I set up a Python virtual environment.

1.  **Created the virtual environment:**
    
    codeBash
    
    ```
    python -m venv venv
    ```
    
2.  **Activated the environment:**
    
    -   Mac/Linux:  source venv/bin/activate
        
    -   Windows:  venv\Scripts\activate
        
3.  **Installed required libraries:**  
    Based on the allowed libraries list, I installed the necessary machine learning and image processing tools:
    
    codeBash
    
    ```
    pip install scikit-learn lightgbm xgboost opencv-python numpy pandas matplotlib seaborn scipy
    ```
    
    (These dependencies were subsequently frozen into requirements.txt)
    

## 3. Baseline Verification

Before writing the new augmentation code, I verified that the existing baseline scripts provided in the repository functioned correctly on my local machine.

1.  **Verified Conversion:** Ran the base image_conversion.py script to ensure PDFs were successfully converting to grayscale images in the data/original_pdfs/ directory.
    
2.  **Verified Classification:** Ran the base classification.py script to confirm the baseline SVM and SGD models could train on the original data and output the expected 97.5–100% accuracy mentioned in the project background.
    

Both scripts executed successfully without pathing errors or missing dependencies.

## 4. GitHub Collaborators

As required by the assignment guidelines, I have granted repository access to the instructional team.

Added delveccj as a collaborator to my GitHub fork.  
-[x] Added AnushkaTi as a collaborator to my GitHub fork.

## 5. Directory Structure Preparation

Finally, I created the necessary empty directories to handle the outputs for the upcoming tasks:

codeBash

```
mkdir -p data/augmented_images
mkdir -p results/confusion_matrices
mkdir -p results/robustness_plots
mkdir -p reports
```
> Written with [StackEdit](https://stackedit.io/).
