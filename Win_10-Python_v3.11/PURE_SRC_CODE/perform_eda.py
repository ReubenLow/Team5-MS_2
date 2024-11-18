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
import textwrap

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
    plot_individual_age_height_weight_relationships(data)
    plot_mean_bmi_by_obesity_level(data)
    create_obesity_summary_table(data)
    plot_combined_mtrans_by_obesity_level(data)
    print(f"EDA graphs saved in {output_folder}")
    


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
    sns.scatterplot(data=data, x="Height", y="Age", hue="Obesity_Level", palette="viridis", alpha=0.7)
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
    sns.scatterplot(data=data, x="Height", y="Weight", hue="Obesity_Level", palette="viridis", alpha=0.7)
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
    sns.scatterplot(data=data, x="Age", y="Weight", hue="Obesity_Level", palette="viridis", alpha=0.7)
    plt.title("Age vs Weight")
    plt.xlabel("Age")
    plt.ylabel("Weight")
    plt.legend(title="Obesity Level", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "age_vs_weight.png"))
    plt.close()

    print("Age vs Weight plot saved.")

    print(f"Scatter plots saved in {output_folder}")

def create_obesity_summary_table(data):
    """
    Creates a summary table for each Obesity_Level_Readable, calculating:
    - Count
    - Mean BMI
    - Standard Deviation (std, calculated manually)
    - Minimum BMI
    - 25th Percentile
    - 50th Percentile (Median)
    - 75th Percentile
    - Maximum BMI
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

    # Calculate the required summary statistics grouped by Obesity_Level_Readable
    grouped = data.groupby("Obesity_Level_Readable")["BMI"]
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

    # Round all numerical values to 3 decimal places
    summary_table = summary_table.round(3)

    # Save the table as a PNG
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=summary_table.values,
        colLabels=summary_table.columns,
        cellLoc="center",
        loc="center",
    )

    # Customize table style: white background with black text
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)  # Adjust size of the table
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_facecolor("white")
        cell.set_text_props(color="black")

    plt.title("Summary Table for Obesity Levels", fontsize=14, pad=10)
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_folder, "obesity_summary_table_readable.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Summary table saved as PNG in {output_file}")

    # Return the summary table as a DataFrame
    return summary_table




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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
    parser.add_argument('--config_file', type=str, default="config.txt", help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config_file)
