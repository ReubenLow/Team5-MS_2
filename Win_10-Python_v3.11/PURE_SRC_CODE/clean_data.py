import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from scipy.stats import shapiro, kstest, norm, probplot, chi2_contingency
import argparse
import configparser
import subprocess
pd.set_option('display.max_rows', None)  # Display all rows

# Function to open config file for review
def open_config_file(config_file):
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' not found. Please make sure it exists.")
        return False
    
    try:
        print(f"Opening configuration file '{config_file}' for review...")
        subprocess.Popen(['open' if os.name == 'posix' else 'start', config_file], shell=True)
        input("Press Enter when you're ready to proceed with data cleaning...")
        return True
    except Exception as e:
        print(f"Failed to open the configuration file: {e}")
        return False

# Function to load paths and suffix from config file
def load_paths_and_suffix(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    paths = {
        "input_folder": config["Paths"].get("input_folder_clean", "../OTH_DATA/training_data"),
        "output_folder": config["Paths"].get("output_folder_clean", "../OTH_DATA/cleaned_data"),
        "cleaned_file_suffix": config["Paths"].get("cleaned_file_suffix", "_v1")
    }
    return paths

# Clean data function
def clean_data(data, drop_columns=None, add_target=False, target_column_name="target"):
    # Drop unnecessary columns
    # If there is extra trailing delimiter, pandas will create an extra column 'Unnamed: 18'
    if 'Unnamed: 18' in data.columns:
        data = data.drop(columns=['Unnamed: 18'])

    # Drop user-specified columns
    if drop_columns:
        data = data.drop(columns=drop_columns, errors='ignore')
        print(f"Dropped columns: {drop_columns}")

    # Handle missing values, fill numerical columns with median
    for column in data.select_dtypes(include=['float64', 'int64']).columns:
        if data[column].isnull().sum() > 0:
            data[column].fillna(data[column].median(), inplace=True)

    # Encoding categorical variables with numbers
    categorical_columns = data.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        data[column] = data[column].astype('category').cat.codes

    # Add an empty target column if requested for dataset with no target column
    if add_target and target_column_name not in data.columns:
        data[target_column_name] = None
        print(f"Added an empty target column: '{target_column_name}'")

    print("Data cleaning complete.")
    return data


def main(config_file="config.txt", selected_file=None):
    # Open config file for user review
    if not open_config_file(config_file):
        print("Exiting due to missing or inaccessible config file.")
        return
    paths = load_paths_and_suffix(config_file)
    input_folder = paths["input_folder"]
    output_folder = paths["output_folder"]
    cleaned_file_suffix = paths["cleaned_file_suffix"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # If no file specified, display available files and prompt for selection
    if selected_file is None:
        files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not files:
            print("No files found in the training_data folder.")
            return
        
        print("\nAvailable files in training_data folder (please wait for a while for files to load):")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input("Enter the number of the file you want to clean: "))
            if 1 <= choice <= len(files):
                selected_file = files[choice - 1]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Please enter a valid number.")
            return

    input_path = os.path.join(input_folder, selected_file)
    data = pd.read_csv(input_path)

    # Ask the user for columns to drop
    user_input = input("Enter the columns you want to drop, separated by commas or press Enter to skip: ")
    drop_columns = ["Patient ID"] 
    if user_input:
        drop_columns += [col.strip() for col in user_input.split(",")]

    # Ask the user if they want to add an empty target column
    add_target = input("Do you want to add an empty target column to this dataset? (yes/no): ").strip().lower() == "yes"
    target_column_name = "target"
    if add_target:
        target_column_name = input("Enter the name of the target column: ")

    # Clean the data
    cleaned_data = clean_data(data, drop_columns=drop_columns, add_target=add_target, target_column_name=target_column_name)

    # Save to the output folder cleaned_data
    output_filename = f"cleaned_{selected_file.split('.')[0]}{cleaned_file_suffix}.csv"
    output_path = os.path.join(output_folder, output_filename)
    cleaned_data.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Clean specific data file in the training_data folder.")
#     parser.add_argument('--input_folder', type=str, default="../OTH_DATA/training_data", help="Folder containing raw data files.")
#     parser.add_argument('--output_folder', type=str, default="../OTH_DATA/cleaned_data", help="Folder to save cleaned data files.")
#     parser.add_argument('--file', type=str, default=None, help="Specific file to clean.")
#     args = parser.parse_args()
#     main(args.input_folder, args.output_folder, args.file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean specific data file in the training_data folder.")
    parser.add_argument('--config_file', type=str, default="config.txt", help="Path to the configuration file.")
    parser.add_argument('--file', type=str, default=None, help="Specific file to clean.")
    args = parser.parse_args()
    main(args.config_file, args.file)