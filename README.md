# ForensicsDetective: Scaling PDF Forensics Through Augmentation and Robustness

**Course:** 510 - Basics of AI  
**Assignment 2:** Forensics Detective - Hero or Zero?  

## Project Overview
This project extends the baseline `ForensicsDetective` research project by introducing domain shifts to evaluate the robustness of machine learning classifiers in PDF provenance detection. By converting PDF binaries to grayscale images, the original project achieved near-perfect accuracy. This repository scales that approach by simulating real-world document complexity (e.g., scanning noise, compression, cropping) and stress-testing four different classifiers.

### Key Features
* **Dataset Augmentation:** Scales the original dataset by 6x using 5 independent image-based augmentations (Gaussian Noise, JPEG Compression, DPI Downsampling, Random Cropping, and Bit-Depth Reduction).
* **Extended Classifiers:** Evaluates the baseline models (SVM, SGD) alongside two newly implemented classifiers (Random Forest, Multi-Layer Perceptron).
* **Robustness Analysis:** Generates performance metrics, robustness curves, and 24 confusion matrices to analyze model degradation under domain shifts.
* **Statistical Testing:** Utilizes McNemar's Test to evaluate the statistical significance of model predictions.

## Repository Structure

```text
Assignment2_YourName/
|-- README.md
|-- SETUP.md
|-- data/
|   |-- original_pdfs/         # Original dataset (PDFs/Images)
|   `-- augmented_images/      # Generated 6x augmented dataset
|-- src/
|   |-- augmentation.py        # Applies the 5 required image augmentations
|   |-- image_conversion.py    # Base PDF-to-image conversion script
|   |-- classification.py      # Defines SVM, SGD, Random Forest, and MLP models
|   |-- analysis.py            # Main pipeline: trains, evaluates, and plots results
|   `-- utils.py               # Helper functions for data loading
|-- results/
|   |-- confusion_matrices/    # 24 generated confusion matrices (PNGs)
|   |-- robustness_plots/      # Robustness degradation curves
|   `-- performance_metrics.csv# Raw accuracy, precision, recall, and F1 scores
|-- reports/
|   `-- final_research_report.pdf # 8-12 page final academic report
`-- requirements.txt           # Python dependencies

```

##Setup and Installation

1.  **Clone the repository:**
    
    codeBash
    
    ``'
        git clone https://github.com/YOUR_USERNAME/ForensicsDetective.git
        cd ForensicsDetective
    ``'
    
2.  **Create a virtual environment (Recommended):**
    
    codeBash
    
    ```
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```
    
3.  **Install dependencies:**
    
    codeBash
    
    ```
    pip install -r requirements.txt
    ```
    
    Primary dependencies include: scikit-learn, opencv-python, numpy, pandas, matplotlib, seaborn, and scipy.
    

(For detailed setup instructions regarding the original fork, please see SETUP.md)

## Running the Pipeline

The project is designed to be run in two main steps:

### 1. Generate the Augmented Dataset

To simulate document complexity, run the augmentation script. This will read the original images, apply the 5 required augmentations independently, and save the new 6x scaled dataset into data/augmented_images/.

codeBash

```
python src/augmentation.py
```

### 2. Run the Comprehensive Analysis

Once the data is generated, execute the analysis pipeline. This script will:

-   Enforce a strict Train/Test split based on the original base images.
    
-   Train the SVM, SGD, Random Forest, and MLP classifiers on the pristine original data.
    
-   Evaluate all models across the 6 conditions (Original + 5 Augmentations).
    
-   Generate and save all confusion matrices, robustness plots, and CSV metrics to the results/ directory.
    
-   Output the McNemar's statistical significance test results to the console.
    

codeBash

```
python src/analysis.py
```

## Key Findings

-   **High Resilience:** All models maintained >99% accuracy under Gaussian Noise and JPEG Compression.
    
-   **Spatial Vulnerability:** Random cropping (1-3%) caused catastrophic failure across all models (SGD dropping to ~54%), proving that flattening 2D images into 1D arrays destroys the spatial alignment required by these classifiers.
    
-   **Algorithmic Collapse:** Bit-depth reduction completely broke the SGD classifier (48.6% accuracy) while the SVM maintained 100% accuracy, highlighting the SGD model's reliance on smooth gradients.
