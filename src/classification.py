from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def get_classifiers():
    """
    Returns a dictionary of classifiers for the pipeline.
    Includes the baseline models from the original repo + 2 new ones for Task 4.
    """
    
    #Keeping seed consistent for the robustness testing later
    SEED = 42

    #Baseline models (from original project requirements)
    svm_clf = SVC(kernel='rbf', probability=True, random_state=SEED)
    sgd_clf = SGDClassifier(loss='log_loss', random_state=SEED)

    # --- TASK 4: Additional Classifiers ---
    
    #1. Random Forest
    #Using n_jobs=-1 because training on the 6x augmented dataset takes a while
    rf_clf = RandomForestClassifier(
        n_estimators=150, 
        max_depth=None,      # let trees grow fully 
        min_samples_split=5, # slight regularization to prevent total overfitting
        n_jobs=-1, 
        random_state=SEED
    )

    #2. Multi-Layer Perceptron (MLP)
    #Since we're dealing with flattened image arrays, the input layer is huge.
    #256 -> 128 bottleneck seems to work okay without running out of memory.
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(256, 128), 
        activation='relu',
        solver='adam',
        max_iter=500,
        early_stopping=True,  #enabled this so it bails out if validation score flatlines
        n_iter_no_change=10,
        random_state=SEED
    )

    #Pack them up for the analysis loop
    models = {
        'SVM': svm_clf,
        'SGD': sgd_clf,
        'Random_Forest': rf_clf,
        'MLP': mlp_clf
    }
    
    return models