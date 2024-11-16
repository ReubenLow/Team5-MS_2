import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
import argparse
from scipy.stats import shapiro, kstest, norm, probplot, chi2_contingency
# from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn import neural_network
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# from sklearn.linear_model import LogisticRegression, Ridge, Perceptron, SGDClassifier
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve, roc_auc_score
from xgboost import XGBClassifier
import joblib
import configparser
import subprocess

# config_file = "config.txt"
def open_config_file(config_file):
    # Check if the config file exists
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' not found. Please make sure it exists.")
        return False
    
    # Open the config file in the default text editor
    try:
        print(f"Opening configuration file '{config_file}' for review...")
        subprocess.Popen(['open' if os.name == 'posix' else 'start', config_file], shell=True)
        input("Press Enter when you're ready to proceed with model training...")
        return True
    except Exception as e:
        print(f"Failed to open the configuration file: {e}")
        return False

# def load_model_params(config_file, model_name):
#     config = configparser.ConfigParser()
#     config.read(config_file)

#     def safe_eval(value):
#         # Attempt to evaluate the value, return as a string if it fails
#         try:
#             return eval(value)
#         except NameError:
#             return value

#     # Return the parsed and safely evaluated parameters
#     return {key: safe_eval(value) for key, value in config[model_name].items()}
# Load model parameters
def load_model_params(config_file, model_name):
    config = configparser.ConfigParser()
    config.read(config_file)
    params = {}
    for key, value in config[model_name].items():
        try:
            # Try to evaluate value as Python literal if possible
            params[key] = eval(value)
        except (NameError, SyntaxError):
            # If eval fails, treat as a string (e.g., "l2" for penalty)
            params[key] = value.strip("'\"")
    return params


# Load paths and suffix
def load_paths_and_suffix(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    paths = {
        "input_folder": config["Paths"].get("input_folder", "../OTH_DATA/cleaned_data"),
        "input_file": config["Paths"].get("input_file", "default.csv"),
        "output_folder": config["Paths"].get("output_folder", "../ML_DATA/model_outputs"),
        "output_file": config["Paths"].get("output_file", "default_model.pkl"),
        "model_name_suffix": config["Paths"].get("model_name_suffix", "_v1")
    }
    return paths


# def train_and_save_models(X, y, model_output, models):
#     if not os.path.exists(model_output):
#         os.makedirs(model_output)

#     # Identify categorical and numerical columns
#     categorical_cols = X.select_dtypes(include=['object', 'category']).columns
#     numeric_cols = X.select_dtypes(include=[np.number]).columns

#     # Create a column transformer to handle both categorical and numeric columns
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), numeric_cols),
#             ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
#         ])

#     for model_name, model in models.items():
#         pipeline = Pipeline([
#             ('preprocessor', preprocessor),  # Preprocess categorical and numeric data
#             ('model', model)
#         ])

#         # Fit the pipeline with preprocessed data
#         pipeline.fit(X, y)

#         # Extract feature names after preprocessing
#         feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
#         pipeline.feature_names = feature_names  # Save feature names to the pipeline

#         # Save the trained model
#         model_path = os.path.join(model_output, f"{model_name}_model.pkl")
#         joblib.dump(pipeline, model_path)
#         print(f"Trained and saved model: {model_name} to {model_path}")

# Train and save selected models
def train_and_save_models(X, y, paths, models):
    if not os.path.exists(paths["output_folder"]):
        os.makedirs(paths["output_folder"])

    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    for model_name, model in models.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        # Fit the pipeline
        pipeline.fit(X, y)

        # Extract feature names after preprocessing
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        pipeline.feature_names = feature_names  # Save feature names to the pipeline

        # Save the trained model with the model name and specified suffix
        model_path = os.path.join(
            paths["output_folder"], 
            f"{model_name}{paths['model_name_suffix']}.pkl"
        )
        joblib.dump(pipeline, model_path)
        print(f"Trained and saved model: {model_name} to {model_path}")

# Select models
def select_models(config_file):
    all_models = {
        "RandomForest": RandomForestClassifier(**load_model_params(config_file, "RandomForest")),
        "AdaBoost": AdaBoostClassifier(**load_model_params(config_file, "AdaBoost")),
        "GradientBoosting": GradientBoostingClassifier(**load_model_params(config_file, "GradientBoosting")),
        "KNeighbors": KNeighborsClassifier(**load_model_params(config_file, "KNeighbors")),
        "SVC": SVC(**load_model_params(config_file, "SVC")),
        "DecisionTree": DecisionTreeClassifier(**load_model_params(config_file, "DecisionTree")),
        "LogisticRegression": LogisticRegression(**load_model_params(config_file, "LogisticRegression")),
        "NaiveBayes": GaussianNB(**load_model_params(config_file, "NaiveBayes")),
        "NeuralNetwork": MLPClassifier(**load_model_params(config_file, "NeuralNetwork")),
        "XGBoost": XGBClassifier(**load_model_params(config_file, "XGBoost"))
    }
    
    # Display model selection prompt
    print("Available models for training:")
    for idx, model_name in enumerate(all_models, start=1):
        print(f"{idx}. {model_name}")
    
    selection = input("Enter the model numbers to train (comma-separated) or 'all' to train all models: ")
    if selection.lower() == 'all':
        selected_models = all_models
    else:
        selected_indices = [int(i.strip()) - 1 for i in selection.split(",")]
        selected_models = {model_name: model for idx, (model_name, model) in enumerate(all_models.items()) if idx in selected_indices}
    
    return selected_models

# def select_features(data):
#     print("Available features:")
#     for idx, column in enumerate(data.columns):
#         print(f"{idx + 1}: {column}")
#     selected = input("Enter the feature numbers to use (comma-separated) or type 'all' to select all: ")
    
#     if selected.lower() == 'all':
#         return data
#     else:
#         selected_indices = [int(i) - 1 for i in selected.split(',')]
#         return data.iloc[:, selected_indices]
def select_features(data):
    print("Available features:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}: {column}")
    selected = input("Enter the feature numbers to use (comma-separated), type 'all' to select all, or press Enter for default (2,3,4,5,6): ")
    
    if selected.lower() == 'all':
        return data
    elif selected.strip() == '':  # Default option if no input
        default_indices = [1, 2, 3, 4, 5]  # 2,3,4,5,6 are indices 1,2,3,4,5 (0-based)
        return data.iloc[:, default_indices]
    else:
        selected_indices = [int(i) - 1 for i in selected.split(',')]
        return data.iloc[:, selected_indices]

def normalize_features(data):
    print("\nAvailable features for normalization:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}: {column}")
    normalize = input("Enter the feature numbers to normalize (comma-separated) or press Enter to skip: ")
    
    if normalize:
        selected_indices = [int(i) - 1 for i in normalize.split(',')]
        selected_features = data.columns[selected_indices]
        for feature in selected_features:
            data[feature] = np.log1p(data[feature]) # Log transformation
            print(f"Applied log transformation on {feature}")
    return data

# def select_training_file(input_folder):
#     files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
#     if not files:
#         print("No CSV files found in the cleaned_data folder.")
#         return None
#     print("Available files for model training:")
#     for idx, file in enumerate(files, 1):
#         print(f"{idx}: {file}")
#     choice = int(input("Select the file number to train the model on: ")) - 1
#     return os.path.join(input_folder, files[choice])

def select_training_file(input_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in the specified input folder.")
        return None
    print("Available files for model training:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}. {file}")
    choice = int(input("Select the file number to use for training: ")) - 1
    return os.path.join(input_folder, files[choice])


# def main(input_folder, model_output, config_file="config.txt"):
def main(config_file="config.txt"):
    if not open_config_file(config_file):
        print("Exiting due to missing or inaccessible config file.")
        return

    # Load paths and model name suffix
    paths = load_paths_and_suffix(config_file)

    # List files in the input folder and prompt user to select one
    data_path = select_training_file(paths["input_folder"])
    if data_path is None:
        print("No valid file selected. Exiting.")
        return

    data = pd.read_csv(data_path)

    # Select features
    data = select_features(data)
    
    # Separate features and target variable
    # target_col = input("Enter the target column (label) by name for training (e.g., 'Survived'): ")
    target_col = "Obesity_Level"
    if target_col not in data.columns:
        raise ValueError(f"The specified target column '{target_col}' does not exist in the data.")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Select models to train
    models = select_models(config_file)

    # Normalize features if needed
    X = normalize_features(X)
    
    # Train and save models
    train_and_save_models(X, y, paths, models)
    print("Training completed and models saved.")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train various models on a selected dataset.")
#     parser.add_argument('--input_folder', type=str, default="../OTH_DATA/cleaned_data", help="Path to the cleaned data folder.")
#     parser.add_argument('--model_output', type=str, default="../ML_DATA/model_outputs", help="Path to save the trained models.")
#     parser.add_argument('--config_file', type=str, default="config.txt", help="Path to the model parameter file.")
    
#     args = parser.parse_args()
#     main(args.input_folder, args.model_output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models on a selected dataset.")
    parser.add_argument('--config_file', type=str, default="config.txt", help="Path to the model parameter file.")
    
    args = parser.parse_args()
    main(args.config_file)
