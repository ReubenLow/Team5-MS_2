import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import shapiro, kstest, norm, probplot, chi2_contingency
import argparse
import configparser
import subprocess
from textwrap import wrap

pd.set_option('display.max_rows', None)  # Display all rows

# Function to open config file for review
def open_config_file(config_file):
    if not os.path.exists(config_file):
        print(f"Configuration file '{config_file}' not found. Please make sure it exists.")
        return False
    
    try:
        print(f"Opening configuration file '{config_file}' for review...")
        subprocess.Popen(['open' if os.name == 'posix' else 'start', config_file], shell=True)
        input("Press Enter when you're ready to proceed with EDA...")
        return True
    except Exception as e:
        print(f"Failed to open the configuration file: {e}")
        return False

# Function to load paths and suffix for EDA from config
def load_eda_paths_and_suffix(config_file):
    config = configparser.ConfigParser()
    config.read(config_file)
    paths = {
        "input_folder": config["Paths"].get("input_folder_eda", "../OTH_DATA/cleaned_data"),
        "output_folder": config["Paths"].get("output_folder_eda", "../EDA_DATA"),
        "eda_file_suffix": config["Paths"].get("eda_file_suffix", "_v1")
    }
    return paths

# Function to calculate BMI column
def calculate_bmi(data):
    """
    Calculates and adds a new column 'BMI' to the DataFrame.

    Parameters:
    - data: pandas DataFrame containing the dataset.

    Returns:
    - data: DataFrame with the new 'BMI' column.
    """
    # Ensure Height is in meters (if needed)
    data['BMI'] = data['Weight'] / (data['Height'] ** 2)
    print("BMI column successfully added.")
    return data

def add_readable_columns(data):
    """
    Adds two new columns to the DataFrame with readable string labels for MTRANS and Obesity_Level.

    Parameters:
    - data: pandas DataFrame containing the dataset with MTRANS and Obesity_Level columns.

    Returns:
    - data: DataFrame with the new readable columns.
    """
    # Define the mapping dictionaries
    mtrans_mapping = {
        0: "Automobile",
        1: "Bike",
        2: "Motorbike",
        3: "Public_Transportation",
        4: "Walking"
    }

    obesity_level_mapping = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }

    # Add new columns with readable strings
    data["MTRANS_Readable"] = data["MTRANS"].map(mtrans_mapping)
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_mapping)

    print("Added readable columns: MTRANS_Readable and Obesity_Level_Readable.")
    return data


def list_files(directory):
    # List all files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    for idx, file in enumerate(files):
        print(f"{idx + 1}. {file}")
    return files

def select_file(files):
    # Prompt user to select a file
    while True:
        try:
            choice = int(input("Enter the number corresponding to the file you want to perform EDA on: "))
            if 1 <= choice <= len(files):
                return files[choice - 1]
            else:
                print("Invalid choice. Please select a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def main(config_file="config.txt"):
    # Load paths and suffix from the configuration file
    paths = load_eda_paths_and_suffix(config_file)
    input_folder = paths["input_folder"]
    output_folder = paths["output_folder"]
    eda_file_suffix = paths["eda_file_suffix"]

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List available CSV files in the input folder
    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    if not files:
        print("No files found in the cleaned data folder.")
        return

    # Display the list of files for user selection
    print("Available files in cleaned data folder:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

    # Prompt the user to select a file
    try:
        choice = int(input("Select the file number to perform EDA on: "))
        if 1 <= choice <= len(files):
            selected_file = files[choice - 1]
        else:
            print("Invalid selection.")
            return
    except ValueError:
        print("Please enter a valid number.")
        return

    # Construct the input path and load the selected file
    input_path = os.path.join(input_folder, selected_file)
    data = pd.read_csv(input_path)
    # Add the BMI column
    data = calculate_bmi(data)
    data = add_readable_columns(data)
    # Call the functions for EDA
    plot_distributions(data)
    plot_box_plots(data)
    plot_obesity_age_distribution(data)
    result_column = "Obesity_Level"
    plot_correlation_heatmap(data, result_column)
    plot_mean_bmi_by_obesity_level(data)
    create_obesity_summary_table(data)
    plot_combined_mtrans_by_obesity_level(data)
    plot_age_categories_by_obesity_level(data)
    #---
    plot_family_history_vs_obesity_level(data)
    plot_individual_age_height_weight_relationships(data)
    plot_favc_vs_obesity_level(data)
    plot_ncp_vs_obesity_level_scatter(data)
    plot_caec_vs_obesity_level(data)
    plot_scc_vs_obesity_level(data)
    plot_faf_vs_obesity_level(data)
    plot_tue_vs_obesity_level(data)
    plot_mtrans_vs_obesity_level(data)
    plot_calc_vs_obesity_level(data)
    plot_calc_vs_simplified_obesity_level(data)
    plot_smoke_vs_obesity_level(data)
    plot_ch2o_vs_obesity_level(data)
    plot_gender_vs_obesity_level(data)
    plot_gender_vs_simplified_obesity(data)
    plot_mean_body_fat_vs_obesity_levels(data)
    plot_sorted_boxplot_body_fat_vs_obesity_levels(data)
    print(f"EDA graphs saved in {output_folder}")

def plot_sorted_boxplot_body_fat_vs_obesity_levels(data):

    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/body_fat_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Calculate median body fat for sorting
    median_body_fat = (
        data.groupby("Obesity_Level_Readable")["Body_Fat"]
        .median()
        .sort_values(ascending=True)
    )

    # Reorder categories based on median body fat
    sorted_categories = median_body_fat.index
    data["Obesity_Level_Readable"] = pd.Categorical(
        data["Obesity_Level_Readable"], categories=sorted_categories, ordered=True
    )

    # Create the sorted box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=data, x="Obesity_Level_Readable", y="Body_Fat", palette="viridis"
    )
    plt.title("Sorted Box Plot: Body Fat Percentage by Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Levels", fontsize=12)
    plt.ylabel("Body Fat Percentage", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "sorted_boxplot_body_fat_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Sorted Box Plot for Body Fat vs Obesity Levels saved in {output_file}")


def plot_mean_body_fat_vs_obesity_levels(data):
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/body_fat_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Calculate the mean body fat for each obesity level and sort in ascending order
    mean_body_fat = (
        data.groupby("Obesity_Level_Readable")["Body_Fat"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    # Create the bar graph
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=mean_body_fat, x="Obesity_Level_Readable", y="Body_Fat", palette="viridis"
    )
    plt.title("Mean Body Fat Percentage vs Obesity Levels (Sorted)", fontsize=16)
    plt.xlabel("Obesity Levels", fontsize=12)
    plt.ylabel("Mean Body Fat Percentage", fontsize=12)
    plt.xticks(rotation=45, ha="right")

    # Add values on top of bars
    for idx, value in enumerate(mean_body_fat["Body_Fat"]):
        plt.text(idx, value + 0.5, f"{value:.2f}", ha="center", fontsize=10, color="black")

    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "mean_body_fat_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Mean Body Fat vs Obesity Levels chart saved in {output_file}")


def plot_gender_vs_simplified_obesity(data):
    """
    Visualizes the relationship between Gender and simplified obesity categories
    (Normal/Underweight and Overweight) using a grouped bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Gender` codes with readable strings
    gender_map = {
        0: "Female",
        1: "Male"
    }
    data["Gender_Readable"] = data["Gender"].map(gender_map)

    # Create a new simplified obesity category
    def simplify_obesity_level(level):
        if level in [0, 1]:  # Insufficient Weight or Normal Weight
            return "Normal/Underweight"
        else:  # Overweight or Obese
            return "Overweight"

    data["Simplified_Obesity_Level"] = data["Obesity_Level"].apply(simplify_obesity_level)

    # Group by Simplified Obesity Level and Gender, then count occurrences
    simplified_counts = data.groupby(["Simplified_Obesity_Level", "Gender_Readable"]).size().unstack(fill_value=0)

    # Plot the grouped bar chart
    simplified_counts.plot(kind="bar", figsize=(10, 6), width=0.8, cmap="viridis")
    plt.title("Gender vs Weight (Overweight and normal/underweight) Categories", fontsize=12)
    plt.xlabel("Obesity Category", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "gender_vs_simplified_obesity.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Gender vs Simplified Obesity Categories chart saved in {output_file}")


def plot_gender_vs_obesity_level(data):
    """
    Visualizes the relationship between Gender and Obesity Levels
    using a grouped bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Gender` codes with readable strings
    gender_map = {
        0: "Female",
        1: "Male"
    }
    data["Gender_Readable"] = data["Gender"].map(gender_map)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Group by Obesity_Level and Gender, then count occurrences
    gender_counts = data.groupby(["Obesity_Level_Readable", "Gender_Readable"]).size().unstack(fill_value=0)

    # Plot the grouped bar chart
    gender_counts.plot(kind="bar", figsize=(12, 8), width=0.8, cmap="viridis")
    plt.title("Gender vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Gender", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "gender_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Gender vs Obesity Levels grouped bar chart saved in {output_file}")


def plot_ch2o_vs_obesity_level(data):
    """
    Visualizes the relationship between CH2O (Daily Water Intake)
    and Obesity Levels using a box plot.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Plot the box plot
    plt.figure(figsize=(12, 8))
    sns.boxplot(
        data=data,
        x="Obesity_Level_Readable",
        y="CH2O",
        palette="viridis"
    )
    plt.title("CH2O (Daily Water Intake) vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("CH2O (Daily Water Intake)", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "ch2o_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"CH2O vs Obesity Levels box plot saved in {output_file}")

def plot_smoke_vs_obesity_level(data):
    """
    Visualizes the relationship between SMOKE (Smoking Habit)
    and Obesity Levels using a stacked bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `SMOKE` codes with readable strings
    smoke_map = {
        0: "No",
        1: "Yes"
    }
    data["SMOKE_Readable"] = data["SMOKE"].map(smoke_map)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Group by Obesity_Level and SMOKE, then count occurrences
    smoke_counts = data.groupby(["Obesity_Level_Readable", "SMOKE_Readable"]).size().unstack(fill_value=0)

    # Plot the stacked bar chart
    smoke_counts.plot(kind="bar", stacked=True, figsize=(12, 8), cmap="viridis")
    plt.title("SMOKE (Smoking Habit) vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="SMOKE (Smoking Habit)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "smoke_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"SMOKE vs Obesity Levels chart saved in {output_file}")


def plot_calc_vs_simplified_obesity_level(data):
    """
    Visualizes the relationship between CALC (Consumption of Alcohol)
    and simplified obesity levels (Normal/Underweight vs Overweight)
    using a grouped bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `CALC` codes with readable strings
    calc_map = {
        0: "Frequently",
        1: "Sometimes",
        2: "No"
    }
    data["CALC_Readable"] = data["CALC"].map(calc_map)

    # Categorize Obesity_Level into two groups: Normal/Underweight vs Overweight
    obesity_category_map = {
        0: "Normal/Underweight",  # Insufficient_Weight
        1: "Normal/Underweight",  # Normal_Weight
        2: "Overweight",          # Obesity_Type_I
        3: "Overweight",          # Obesity_Type_II
        4: "Overweight",          # Obesity_Type_III
        5: "Overweight",          # Overweight_Level_I
        6: "Overweight"           # Overweight_Level_II
    }
    data["Simplified_Obesity_Level"] = data["Obesity_Level"].map(obesity_category_map)

    # Group by Simplified Obesity Level and CALC, then count occurrences
    calc_counts = data.groupby(["Simplified_Obesity_Level", "CALC_Readable"]).size().unstack(fill_value=0)

    # Plot the grouped bar chart
    calc_counts.plot(kind="bar", figsize=(10, 6), width=0.8, cmap="viridis")

    # Customize the plot
    plt.title("CALC (Consumption of Alcohol) vs Overweight and (underweight/normal weight)", fontsize=16)
    plt.xlabel("Overweight and (underweight/normal weight)", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=0, ha="center")
    plt.legend(title="CALC (Alcohol Consumption)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "calc_vs_simplified_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"CALC vs Simplified Obesity Levels chart saved in {output_file}")



def plot_calc_vs_obesity_level(data):
    """
    Visualizes the relationship between CALC (Consumption of Alcohol)
    and Obesity Levels using a grouped bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `CALC` codes with readable strings
    calc_map = {
        0: "Frequently",
        1: "Sometimes",
        2: "No"
    }
    data["CALC_Readable"] = data["CALC"].map(calc_map)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Group by Obesity_Level and CALC, then count occurrences
    calc_counts = data.groupby(["Obesity_Level_Readable", "CALC_Readable"]).size().unstack(fill_value=0)

    # Plot the grouped bar chart
    calc_counts.plot(kind="bar", figsize=(12, 8), width=0.8, cmap="viridis")

    # Customize the plot
    plt.title("CALC (Consumption of Alcohol) vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="CALC (Alcohol Consumption)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "calc_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"CALC vs Obesity Levels chart saved in {output_file}")



def plot_mtrans_vs_obesity_level(data):
    """
    Visualizes the relationship between MTRANS (Mode of Transportation)
    and Obesity Levels using a stacked bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `MTRANS` codes with readable strings
    mtrans_map = {
        0: "Automobile",
        1: "Bike",
        2: "Motorbike",
        3: "Public_Transportation",
        4: "Walking"
    }
    data["MTRANS_Readable"] = data["MTRANS"].map(mtrans_map)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Group by Obesity_Level and MTRANS, then count occurrences
    mtrans_counts = data.groupby(["Obesity_Level_Readable", "MTRANS_Readable"]).size().unstack(fill_value=0)

    # Plot the stacked bar chart
    mtrans_counts.plot(kind="bar", stacked=True, figsize=(12, 8), cmap="viridis")
    plt.title("MTRANS (Mode of Transportation) vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="MTRANS (Mode of Transportation)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "mtrans_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"MTRANS vs Obesity Levels chart saved in {output_file}")


def plot_tue_vs_obesity_level(data):
    """
    Visualizes the relationship between TUE (Time of Physical Activity per Day)
    and Obesity Levels using a scatter plot.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/scatter_plots")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x="TUE",
        y="Obesity_Level_Readable",
        hue="Obesity_Level_Readable",
        palette="viridis",
        alpha=0.7,
        s=100
    )
    plt.title("TUE (Time using technological devices per Day) vs Obesity Levels", fontsize=16)
    plt.xlabel("TUE (Time using technological devices per Day)", fontsize=12)
    plt.ylabel("Obesity Level", fontsize=12)
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "tue_vs_obesity_levels_scatter.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Scatter plot saved in {output_file}")



def plot_faf_vs_obesity_level(data):
    """
    Visualizes the relationship between FAF (Physical Activity Frequency)
    and Obesity Levels using a scatter plot.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/scatter_plots")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x="FAF",
        y="Obesity_Level_Readable",
        hue="Obesity_Level_Readable",
        palette="viridis",
        alpha=0.7,
        s=100
    )
    plt.title("FAF (Physical Activity Frequency) vs Obesity Levels", fontsize=16)
    plt.xlabel("FAF (Physical Activity Frequency)", fontsize=12)
    plt.ylabel("Obesity Level", fontsize=12)
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "faf_vs_obesity_levels_scatter.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Scatter plot saved in {output_file}")


def plot_scc_vs_obesity_level(data):
    """
    Visualizes the relationship between SCC (Calories Monitoring) and Obesity Levels
    using a stacked bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `SCC` codes with readable strings
    scc_map = {
        0: "No",
        1: "Yes"
    }
    data["SCC_Readable"] = data["SCC"].map(scc_map)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Group by Obesity_Level and SCC, then count occurrences
    scc_counts = data.groupby(["Obesity_Level_Readable", "SCC_Readable"]).size().unstack(fill_value=0)

    # Plot the stacked bar chart
    scc_counts.plot(kind="bar", stacked=True, figsize=(12, 8), cmap="viridis")
    plt.title("SCC (Calories Monitoring) vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="SCC (Calories Monitoring)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "scc_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"SCC vs Obesity Levels chart saved in {output_file}")


def plot_caec_vs_obesity_level(data):
    """
    Visualizes the relationship between CAEC (Consumption of food between meals) 
    and Obesity Levels using a stacked bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `CAEC` codes with readable strings
    caec_map = {
        0: "Always",
        1: "Frequently",
        2: "Sometimes",
        3: "No"
    }
    data["CAEC_Readable"] = data["CAEC"].map(caec_map)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Group by Obesity_Level and CAEC, then count occurrences
    caec_counts = data.groupby(["Obesity_Level_Readable", "CAEC_Readable"]).size().unstack(fill_value=0)

    # Plot the stacked bar chart
    caec_counts.plot(kind="bar", stacked=True, figsize=(12, 8), cmap="viridis")
    plt.title("CAEC vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="CAEC (Consumption of Food Between Meals)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "caec_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"CAEC vs Obesity Levels chart saved in {output_file}")



def plot_ncp_vs_obesity_level_scatter(data):
    """
    Visualizes the relationship between Number of Meals (NCP) and obesity levels
    using a scatter plot.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }

    # Map the Obesity_Level to readable strings
    data["Obesity_Level_Readable"] = data["Obesity_Level"].map(obesity_level_map)

    # Plot the scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=data,
        x="NCP",  # Number of Meals
        y="Obesity_Level_Readable",  # Obesity Level
        hue="Obesity_Level_Readable",  # Color points by Obesity Level
        palette="viridis",
        alpha=0.7,
        marker="o",
        s=100  # Marker size
    )

    plt.title("Number of Meals (NCP) vs Obesity Levels", fontsize=16)
    plt.xlabel("Number of Meals (NCP)", fontsize=12)
    plt.ylabel("Obesity Level", fontsize=12)
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "ncp_vs_obesity_levels_scatter.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"NCP vs Obesity Levels scatter plot saved in {output_file}")


    
def plot_individual_age_height_weight_relationships(data):
    """
    Plots scatter plots for:
    1. Height vs Age
    2. Height vs Weight
    3. Age vs Weight
    Each scatter plot is saved as a separate file.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/individual_relationship_plots")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Scatter plot: Height vs Age
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="Height",
        y="Age",
        hue="Obesity_Level_Readable",  # Use readable strings for legend
        palette="viridis",
        alpha=0.7
    )
    plt.title("Height vs Age")
    plt.xlabel("Height")
    plt.ylabel("Age")
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "height_vs_age.png"))
    plt.close()

    print("Height vs Age plot saved.")

    # Scatter plot: Height vs Weight
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="Height",
        y="Weight",
        hue="Obesity_Level_Readable",  # Use readable strings for legend
        palette="viridis",
        alpha=0.7
    )
    plt.title("Height vs Weight")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "height_vs_weight.png"))
    plt.close()

    print("Height vs Weight plot saved.")

    # Scatter plot: Age vs Weight
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=data,
        x="Age",
        y="Weight",
        hue="Obesity_Level_Readable",  # Use readable strings for legend
        palette="viridis",
        alpha=0.7
    )
    plt.title("Age vs Weight")
    plt.xlabel("Age")
    plt.ylabel("Weight")
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "age_vs_weight.png"))
    plt.close()

    print("Age vs Weight plot saved.")

    print(f"Scatter plots saved in {output_folder}")


def plot_box_plots(data):
    """
    Plots box plots for Weight, Age, and grouped box plots for Weight vs Age Group.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/box plots")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Box plot for Weight
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='Weight', color="skyblue")
    plt.title('Box Plot for Weight')
    plt.xlabel('Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'weight_box_plot.png'))
    plt.close()

    # Box plot for Age
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x='Age', color="lightgreen")
    plt.title('Box Plot for Age')
    plt.xlabel('Age')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'age_box_plot.png'))
    plt.close()

    # Create Age_Group for grouped box plot
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '66+'])

    # Grouped box plot: Weight vs Age_Group
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Age_Group', y='Weight', palette="pastel")
    plt.title('Box Plot of Weight Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'weight_age_group_box_plot.png'))
    plt.close()

    print(f"Box plots saved in {output_folder}")


def plot_combined_mtrans_by_obesity_level(data):
    """
    Plots a combined bar graph for all MTRANS types across each Obesity_Level.
    Each group of bars represents the distribution of MTRANS types for an Obesity_Level,
    using the 'MTRANS_Readable' and 'Obesity_Level_Readable' columns for better readability.

    Parameters:
    - data: pandas DataFrame containing the dataset with 'MTRANS_Readable' and 'Obesity_Level_Readable' columns.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/mtrans_combined_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Count the number of occurrences of MTRANS_Readable types for each Obesity_Level_Readable
    mtrans_counts = data.groupby(["Obesity_Level_Readable", "MTRANS_Readable"]).size().unstack(fill_value=0)

    # Plot the grouped bar chart
    plt.figure(figsize=(12, 8))
    mtrans_counts.plot(kind="bar", figsize=(12, 8), width=0.8)

    # Customize the plot
    plt.title("Distribution of MTRANS by Obesity Level", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Mode of Transportation (MTRANS)", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "combined_mtrans_distribution_readable.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Combined bar graph saved in {output_file}")




def plot_obesity_age_distribution(data):
    """
    Plots the distribution of obesity levels across age groups using
    stacked and grouped bar charts.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Get the current script directory and define a relative output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/distribution plots")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Create Age_Group for grouping
    data['Age_Group'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '66+'])

    # Calculate counts for each Age_Group and Obesity_Level
    obesity_age_counts = data.groupby(['Age_Group', 'Obesity_Level']).size().unstack(fill_value=0)

    # Stacked Bar Chart
    obesity_age_counts.plot(kind='bar', stacked=True, figsize=(10, 6), cmap='viridis')
    plt.title('Stacked Bar Chart of Obesity Levels Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(title='Obesity Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'stacked_bar_chart_obesity_age.png'))
    plt.close()

    # Grouped Bar Chart
    obesity_age_counts.plot(kind='bar', stacked=False, figsize=(10, 6), width=0.8, cmap='viridis')
    plt.title('Grouped Bar Chart of Obesity Levels Across Age Groups')
    plt.xlabel('Age Group')
    plt.ylabel('Count')
    plt.legend(title='Obesity Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'grouped_bar_chart_obesity_age.png'))
    plt.close()

    print(f"Distribution graphs saved in {output_folder}")


def plot_correlation_heatmap(data, result_column):
    """
    Plots a correlation heatmap of all numerical features against the result column.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    - result_column: The name of the result/target column to analyze correlations.
    - output_folder: Folder to save the heatmap.
    """

    # Get the current script directory and define a relative output folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_heat_map")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Compute correlations
    corr_matrix = data.corr()

    # Sort by the target/result column for better focus
    if result_column in corr_matrix:
        corr_with_target = corr_matrix[result_column].sort_values(ascending=False)
    else:
        print(f"Warning: '{result_column}' not found in correlation matrix.")
        corr_with_target = None

    # Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'correlation_heatmap.png'))
    plt.close()

    print(f"Correlation heatmap saved in {output_folder}")

    # Show top correlations with the result variable
    if corr_with_target is not None:
        print("\nTop correlations with result variable:")
        print(corr_with_target)



def create_obesity_summary_table(data):
    """
    Creates a summary table for each Obesity_Level with readable strings as row values, calculating:
    - Count
    - Mean Weight
    - Standard Deviation (std, calculated manually)
    - Minimum Weight
    - 25th Percentile
    - 50th Percentile (Median)
    - 75th Percentile
    - Maximum Weight
    Saves the table as a PNG file with white background and black text.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """


    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/obesity_summary_table")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the required summary statistics grouped by Obesity_Level
    grouped = data.groupby("Obesity_Level")["Weight"]
    summary_table = pd.DataFrame({
        "count": grouped.size(),
        "mean": grouped.mean(),
        "std": grouped.apply(lambda x: ((x - x.mean()) ** 2).sum() / (len(x) - 1)) ** 0.5,
        "min": grouped.min(),
        "percentile_25": grouped.quantile(0.25),
        "percentile_50": grouped.median(),
        "percentile_75": grouped.quantile(0.75),
        "max": grouped.max()
    }).reset_index()

    # Replace `Obesity_Level` with readable strings in the summary table
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    summary_table["Obesity_Level"] = summary_table["Obesity_Level"].map(obesity_level_map)

    # Round all numerical values to 3 decimal places
    summary_table = summary_table.round(3)

    # Wrap column headers for better readability
    col_labels = [
        "\n".join(wrap(col, width=10)) for col in summary_table.columns
    ]

    # Save the table as a PNG
    fig, ax = plt.subplots(figsize=(15, 8))  # Keep image size
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=summary_table.values,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )

    # Customize column widths and cell heights
    for (row, col), cell in table.get_celld().items():
        if col == 0:  # Keep Obesity_Level column wide
            cell.set_width(0.2)  # Set wider width for this column
            cell.set_height(0.11)  # Adjust height
        else:  # Reduce the width of other columns
            cell.set_width(0.07)
            cell.set_height(0.11)

    # Customize table style: white background with black text
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Adjust font size
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_facecolor("white")
        cell.set_text_props(color="black")

    plt.title("Summary Table for Obesity Levels (Based on Weight)", fontsize=16, pad=20)  # Adjust title size and padding
    plt.tight_layout()

    # Save the plot with adjusted bounding box for better layout
    output_file = os.path.join(output_folder, "obesity_summary_table_weight.png")
    plt.savefig(output_file, bbox_inches="tight", dpi=300)  # High resolution and adjusted layout
    plt.close()

    print(f"Summary table saved as PNG in {output_file}")

    # Return the summary table as a DataFrame
    return summary_table



def plot_favc_vs_obesity_level(data):
    """
    Visualizes the correlation between FAVC and obesity levels using a stacked bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare the data: Group by Obesity_Level and FAVC
    favc_counts = data.groupby(["Obesity_Level", "FAVC"]).size().unstack(fill_value=0)

    # Replace `FAVC` codes with readable strings
    favc_counts.rename(columns={0: "No", 1: "Yes"}, inplace=True)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    favc_counts.index = favc_counts.index.map(obesity_level_map)

    # Plot the stacked bar chart
    favc_counts.plot(kind="bar", stacked=True, figsize=(12, 8), color=["skyblue", "orange"])
    plt.title("FAVC vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="FAVC (High-Calorie Food Consumption)", loc="upper right")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "favc_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"FAVC vs Obesity Levels chart saved in {output_file}")






def plot_mean_bmi_by_obesity_level(data):
    """
    Calculates the mean BMI for each Obesity_Level_Readable and plots a sorted bar graph.
    Includes mean BMI values as annotations on the bars.

    Parameters:
    - data: pandas DataFrame containing the dataset with the 'Obesity_Level_Readable' column.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/bmi_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate mean BMI for each Obesity_Level_Readable and sort by mean BMI
    mean_bmi = (
        data.groupby("Obesity_Level_Readable")["BMI"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    # Plot the bar graph
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        x="Obesity_Level_Readable", y="BMI", data=mean_bmi, palette="viridis", order=mean_bmi["Obesity_Level_Readable"]
    )

    # Add annotations on top of the bars
    for idx, value in enumerate(mean_bmi["BMI"]):
        ax.text(idx, value + 0.2, f"{value:.2f}", color="black", ha="center", fontsize=10)

    # Set titles and labels
    plt.title("Mean BMI by Obesity Level (Sorted)", fontsize=14)
    plt.xlabel("Obesity Level")
    plt.ylabel("Mean BMI")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "mean_bmi_by_obesity_level_readable_sorted.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Mean BMI bar graph saved in {output_file}")



def map_columns_to_readable_strings(data):
    """
    Maps integer-coded columns to readable string labels.
    Adds new columns with '_Readable' suffix for each mapped column.

    Parameters:
    - data: pandas DataFrame containing the dataset.

    Returns:
    - data: DataFrame with new '_Readable' columns.
    """
    # Define the mapping dictionaries
    mappings = {
        "Gender": {0: "Female", 1: "Male"},
        "fam_hist_over-wt": {0: "no", 1: "yes"},
        "FAVC": {0: "no", 1: "yes"},
        "CAEC": {0: "Always", 1: "Frequently", 2: "Sometimes", 3: "no"},
        "SMOKE": {0: "no", 1: "yes"},
        "SCC": {0: "no", 1: "yes"},
        "CALC": {0: "Frequently", 1: "Sometimes", 2: "no"}
    }

    # Apply mappings to create new readable columns
    for column, mapping in mappings.items():
        if column in data.columns:
            readable_column = f"{column}_Readable"
            data[readable_column] = data[column].map(mapping)

    print("Readable string columns added to the dataset.")
    return data


def plot_distributions(data):
    """
    Plots the distributions of specified columns using readable string labels.
    Increases figure width for columns with long x-axis labels.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/distribution_plots")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Columns to plot (use readable versions if available)
    columns = [
        "Gender_Readable", "Age", "Height", "Weight", "fam_hist_over-wt_Readable",
        "FAVC_Readable", "FCVC", "NCP", "CAEC_Readable", "SMOKE_Readable",
        "CH2O", "SCC_Readable", "FAF", "TUE", "CALC_Readable", 
        "MTRANS_Readable", "Obesity_Level_Readable"
    ]

    for column in columns:
        if column not in data.columns:
            print(f"Skipping column '{column}' as it is not present in the dataset.")
            continue

        # Adjust figure width for long labels
        figure_width = 12 if column == "Obesity_Level_Readable" else 8
        plt.figure(figsize=(figure_width, 6))
        
        # Choose appropriate plot type based on column type
        if data[column].dtype == 'object' or data[column].nunique() < 10:
            # Bar plot for categorical variables
            sns.countplot(data=data, x=column, palette="pastel")
            plt.xticks(rotation=45)
            plt.title(f'Distribution of {column.replace("_Readable", "")}')
            plt.ylabel('Count')

        else:
            # Histogram for numerical variables
            sns.histplot(data[column], kde=True, bins=30, color="skyblue", edgecolor="black")
            plt.title(f'Distribution of {column}')
            plt.ylabel('Density')
        
        plt.xlabel(column.replace("_Readable", ""))
        plt.tight_layout()
        
        # Save plot to the output folder
        plt.savefig(os.path.join(output_folder, f'{column}_distribution.png'))
        plt.close()

    print(f"Distribution plots saved in {output_folder}")


def plot_age_categories_by_obesity_level(data):
    """
    Categorizes the 'Age' column into predefined categories and plots the distribution
    of these categories as bar graphs for each Obesity_Level_Readable.

    Parameters:
    - data: pandas DataFrame containing the dataset with 'Age' and 'Obesity_Level_Readable' columns.
    """

    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/age_categories_by_obesity")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Categorize 'Age' into predefined categories
    age_bins = [0, 12, 18, 35, 60, 100]  # Boundaries for age groups
    age_labels = ["Children", "Teenage", "Young Adults", "Adults", "Elderly"]
    data["Age_Category"] = pd.cut(data["Age"], bins=age_bins, labels=age_labels, right=False)

    # Count occurrences of age categories for each Obesity_Level_Readable
    age_category_counts = data.groupby(["Obesity_Level_Readable", "Age_Category"]).size().unstack(fill_value=0)

    # Plot the grouped bar chart
    plt.figure(figsize=(12, 8))
    age_category_counts.plot(kind="bar", figsize=(12, 8), width=0.8, cmap="viridis")

    # Customize the plot
    plt.title("Distribution of Age Categories by Obesity Level", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Age Categories", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "age_categories_by_obesity_level.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Bar graph saved in {output_file}")


def plot_family_history_vs_obesity_level(data):
    """
    Visualizes the correlation between family history of being overweight 
    and obesity levels using a stacked bar chart.

    Parameters:
    - data: pandas DataFrame containing the dataset.
    """
    # Dynamically set the output folder relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "../EDA_DATA/correlation_analysis")

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Prepare the data: Group by Obesity_Level and fam_hist_over-wt
    family_history_counts = data.groupby(["Obesity_Level", "fam_hist_over-wt"]).size().unstack(fill_value=0)

    # Replace `fam_hist_over-wt` codes with readable strings
    family_history_counts.rename(columns={0: "No", 1: "Yes"}, inplace=True)

    # Replace `Obesity_Level` codes with readable strings
    obesity_level_map = {
        0: "Insufficient_Weight",
        1: "Normal_Weight",
        2: "Obesity_Type_I",
        3: "Obesity_Type_II",
        4: "Obesity_Type_III",
        5: "Overweight_Level_I",
        6: "Overweight_Level_II"
    }
    family_history_counts.index = family_history_counts.index.map(obesity_level_map)

    # Plot the stacked bar chart
    family_history_counts.plot(kind="bar", stacked=True, figsize=(12, 8), color=["skyblue", "orange"])
    plt.title("Family History vs Obesity Levels", fontsize=16)
    plt.xlabel("Obesity Level", fontsize=12)
    plt.ylabel("Number of People", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Family History", loc="upper right")
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "family_history_vs_obesity_levels.png")
    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"Family history vs obesity levels chart saved in {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
    parser.add_argument('--config_file', type=str, default="config.txt", help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config_file)
