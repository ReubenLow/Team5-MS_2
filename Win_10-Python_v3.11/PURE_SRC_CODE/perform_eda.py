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

# Clean data function
# def clean_data(data):
#     # Extract title from name
#     if 'Name' in data.columns:
#         data['Title'] = data['Name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

#     # Drop irrelevant columns
#     data = data.drop(['Passenger ID', 'Ticket Number', 'Cabin', 'Name'], axis=1, errors='ignore')

#     if 'Survived' in data.columns:
#         data['Survived'] = data['Survived'].replace({'Yes': 1, 'No': 0})  # Replace yes/no with 1/0
#         data['Survived'] = data['Survived'].astype(int)  # Ensure Survived is integer

#     # Remove $ sign and commas from 'Passenger Fare' and convert to float
#     data['Passenger Fare'] = data['Passenger Fare'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
#     data['Passenger Fare'] = data['Passenger Fare'].round(2)

#     # Fill zero ages with nan to be processed later
#     data['Age'] = data['Age'].replace(0, np.nan)

#     title_age_medians = data.groupby('Title')['Age'].median().round().to_dict()

#     # Define name titles
#     title_age_medians = {
#         'Mr': data[data['Title'] == 'Mr']['Age'].median(),
#         'Mrs': data[data['Title'] == 'Mrs']['Age'].median(),
#         'Miss': data[data['Title'] == 'Miss']['Age'].median(),
#         'Master': data[data['Title'] == 'Master']['Age'].median(),
#         'Dr': data[data['Title'] == 'Mr']['Age'].median(),  # Assume Dr is adult male
#         'Sir': data[data['Title'] == 'Mr']['Age'].median(),
#         'Lady': data[data['Title'] == 'Mrs']['Age'].median(),
#         'Ms': data[data['Title'] == 'Miss']['Age'].median(),
#         'Mme': data[data['Title'] == 'Mrs']['Age'].median(),
#         'Mlle': data[data['Title'] == 'Miss']['Age'].median()
#     }
    

#     # Fill missing ages based on title medians, ensure that age is at least 1 since original train data has weird ages like 0.73
#     data['Age'] = data.apply(lambda row: max(1, int(np.ceil(title_age_medians[row['Title']]))) if pd.isnull(row['Age']) else max(1, int(np.ceil(row['Age']))), axis=1)

#     # Create age groups after filling missing ages
#     data['Age Group'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '66+'])

#     # Handle missing data
#     imputer = SimpleImputer(strategy='mean')
#     data['Passenger Fare'] = imputer.fit_transform(data[['Passenger Fare']])

#     # Retain original Ticket Class for grouping
#     original_ticket_class = data['Ticket Class']
#     original_embarkation = data['Embarkation Country']

#     # Encode categorical columns with OneHotEncoder
#     categorical_cols = ['Ticket Class', 'Embarkation Country']
#     encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
#     encoded_features.columns = encoder.get_feature_names_out(categorical_cols)

#     # Drop original categorical columns and add encoded features
#     data = data.drop(categorical_cols, axis=1)
#     data = pd.concat([data, encoded_features], axis=1)

#     # Add the original 'Ticket Class' back for grouping
#     data['Ticket Class'] = original_ticket_class
#     data['Embarkation Country'] = original_embarkation

#     return data

# Function to plot survival by different groupings
def plot_survival_by_group(data, group_by, output_folder, chart_type='bar'):
    if chart_type == 'bar':
        # Group by survival and the specified column
        group_counts = data.groupby(['Survived', group_by]).size().unstack()

        # Plot stacked bar chart
        ax = group_counts.plot(kind='bar', stacked=True, color=['#4daf4a', '#377eb8', '#ff7f00'], figsize=(8, 6))
        plt.title(f'Survival rate by {group_by}')
        plt.ylabel('Number of Passengers')
        plt.xticks([0, 1], ['Died', 'Survived'], rotation=0)
        plt.legend(title=group_by, loc='best')
         # Add annotations on bars to show counts
        # for p in ax.patches:
        #     ax.annotate(f'{int(p.get_height())}', 
        #                 (p.get_x() + p.get_width() / 2., p.get_height()), 
        #                 ha='center', va='center', 
        #                 xytext=(0, 10),  # Offset text
        #                 textcoords='offset points')
        plt.savefig(os.path.join(output_folder, f'survival_by_{group_by}.png'))
        plt.close()

    elif chart_type == 'pie':
        # Generate pie chart for total survivors and deaths
        survival_counts = data['Survived'].value_counts()
        total_survivors = survival_counts.loc[1]
        total_deaths = survival_counts.loc[0]

        # Plot pie chart
        labels = ['Survivors', 'Deaths']
        sizes = [total_survivors, total_deaths]
        colors = ['#66b3ff', '#ff6666']
        explode = (0.1, 0)  # explode 1st slice (Survivors)

        def autopct_format(pct, total_values):
            absolute = int(round(pct / 100. * np.sum(total_values)))
            return f'{pct:.1f}%\n({absolute})'

        plt.figure(figsize=(7, 7))
        plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct=lambda pct: autopct_format(pct, sizes), shadow=True, startangle=90)
        plt.title('Survival Distribution')
        plt.savefig(os.path.join(output_folder, f'survival_distribution_pie.png'))
        plt.close()

# Function to print survival and death numbers by ticket class
def print_survival_death_by_class_and_gender(data):
    # Group by 'Ticket Class' and 'Survived'
    survival_death_by_class = data.groupby(['Ticket Class', 'Survived']).size().unstack(fill_value=0)
    print("\nSurvival and Death Numbers by Ticket Class:")
    print(survival_death_by_class)

    # Group by 'Gender' and 'Survived'
    survival_death_by_gender = data.groupby(['Gender', 'Survived']).size().unstack(fill_value=0)
    print("\nSurvival and Death Numbers by Gender:")
    print(survival_death_by_gender)

# Function to plot survival rate by gender within each class
def plot_survival_rate_by_gender_and_class(data, output_folder):
    # Calculate survival rate by 'Ticket Class' and 'Gender'
    survival_rate = data.groupby(['Ticket Class', 'Gender'])['Survived'].mean().unstack()
    
    # Plot the bar chart
    survival_rate.plot(kind='bar', figsize=(8, 6), stacked=False, color=['#377eb8', '#ff7f00'])  # Male: blue, Female: orange
    plt.title('Survival rate by gender and class')
    plt.xlabel('Ticket Class')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.legend(title='Gender')

    # Save the chart
    plt.savefig(os.path.join(output_folder, 'survival_rate_by_gender_and_class.png'))
    plt.close()

    # Print the survival rate table
    print("\nSurvival Rate by Gender and Class:")
    print(survival_rate)

    # Print the number of male and female passengers in each class
    gender_count_by_class = data.groupby(['Ticket Class', 'Gender']).size().unstack()
    print("\nNumber of Male and Female Passengers by Class:")
    print(gender_count_by_class)

def plot_age_pyramid(data, output_folder):
    if 'Age Group' not in data.columns or 'Gender' not in data.columns:
        print("Error: 'Age Group' or 'Gender' column is missing in the data.")
        return

    # Group by 'Survived', 'Age Group', and 'Gender', and fill missing groups with 0
    age_pyramid = data.groupby(['Survived', 'Age Group', 'Gender'], observed=True).size().unstack(level=['Age Group', 'Gender']).fillna(0)
    print("Grouped Data (age_pyramid):\n", age_pyramid)

    # Create a pyramid plot
    fig, ax = plt.subplots(figsize=(10, 6))
    age_groups = age_pyramid.columns.get_level_values(0).unique()

    # Plot each gender's survival status by age group
    for survival_status, color_map in zip([0, 1], [['#377eb8', '#ff7f00'], ['#4daf4a', '#ff7f00']]):
        for gender, color in zip(['female', 'male'], color_map):
            for age_group in age_groups:
                # Check if the specific column exists in age_pyramid
                if (age_group, gender) in age_pyramid.columns:
                    bar_data = -age_pyramid.loc[survival_status, (age_group, gender)] if survival_status == 0 else age_pyramid.loc[survival_status, (age_group, gender)]
                    ax.barh(age_group, bar_data, color=color, label=f"{gender.capitalize()} ({'Died' if survival_status == 0 else 'Survived'})", alpha=0.6 if survival_status == 1 else 1)

    # Adjustments for labels and layout
    ax.set_xlabel("Count")
    ax.set_title("Survival by Age Group and Gender (Pyramid Plot)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'age_pyramid_survival.png'))
    plt.close()
    # plt.show()

def plot_survival_rate_by_age_group(data, output_folder):
    # Calculate survival rate by age group
    survival_rate_by_age = data.groupby('Age Group')['Survived'].mean()

    # Plot survival rate by age group
    plt.figure(figsize=(8, 5))
    survival_rate_by_age.plot(kind='bar', color='#4daf4a', alpha=0.7)
    plt.title('Survival Rate by Age Group')
    plt.xlabel('Age Group')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'survival_rate_by_age_group.png'))
    plt.close()

def plot_age_density(data, output_folder):
    # Plot kernel density estimation of age for survivors vs non-survivors
    plt.figure(figsize=(8, 5))
    sns.kdeplot(data=data[data['Survived'] == 1], x='Age', fill=True, color='#4daf4a', label='Survived', alpha=0.5)
    sns.kdeplot(data=data[data['Survived'] == 0], x='Age', fill=True, color='#ff6666', label='Did Not Survive', alpha=0.5)
    plt.title('Age Density Plot by Survival')
    plt.xlabel('Age')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'age_density_survival.png'))
    plt.close()

# Main function to analyze survival by age
def analyze_survival_by_age(data, output_folder):
    # Ensure Age Group is defined (if not, create it based on bins)
    if 'Age Group' not in data.columns:
        data['Age Group'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '66+'])

    # Plot pyramid plot
    plot_age_pyramid(data, output_folder)

    # Plot survival rate by age group
    plot_survival_rate_by_age_group(data, output_folder)

    # Plot age density by survival
    plot_age_density(data, output_folder)

def analyze_survival_by_ticket_price(data, output_folder):
    max_price = data['Passenger Fare'].max()
    quartile_bins = [0, 0.25 * max_price, 0.5 * max_price, 0.75 * max_price, max_price]
    bin_labels = ['0-25%', '25-50%', '50-75%', '75-100%']

    # Use pd.cut to categorize ticket prices into quartiles
    data['Ticket Price Bin'] = pd.cut(data['Passenger Fare'], bins=quartile_bins, labels=bin_labels, include_lowest=True)

    # Calculate survival rate by ticket price bin
    survival_rate_by_price = data.groupby('Ticket Price Bin')['Survived'].mean()

    # Plot survival rate by ticket price bin
    plt.figure(figsize=(8, 5))
    survival_rate_by_price.plot(kind='bar', color='lightgreen', alpha=0.7)
    plt.title('Survival Rate by Ticket Price Quartile')
    plt.xlabel('Ticket Price Quartile (of Max Price)')
    plt.ylabel('Survival Rate')
    plt.xticks(rotation=0)

    # Display the price ranges on the x-axis labels
    for i, label in enumerate(bin_labels):
        price_range = f"({quartile_bins[i]:.2f} - {quartile_bins[i + 1]:.2f})"
        plt.text(i, survival_rate_by_price[i], f"{label}\n{price_range}", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'survival_rate_by_ticket_price.png'))
    plt.close()

    print("\nSurvival Rate by Ticket Price Bin:")
    print(survival_rate_by_price)

def analyze_fare_distribution(data, output_folder):
    fares = data['Passenger Fare'].dropna()  # Drop missing fares

    # Plot histogram and kde
    plt.figure(figsize=(10, 6))
    sns.histplot(fares, kde=True, bins=30, color="skyblue", edgecolor="black", stat="density")
    plt.title("Fare Distribution with KDE Overlay")
    plt.xlabel("Passenger Fare")
    plt.ylabel("Density")
    plt.savefig(f"{output_folder}/fare_distribution.png")
    plt.close()

    # Q-q Plot
    plt.figure(figsize=(6, 6))
    probplot(fares, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Fare Distribution")
    plt.savefig(f"{output_folder}/fare_qq_plot.png")
    plt.close()

    # Shapiro wilk Test
    shapiro_stat, shapiro_p_value = shapiro(fares)
    print(f"Shapiro wilk Test: Statistic={shapiro_stat:.4f}, p-value={shapiro_p_value:.4f}")

    # Kolmogorov smirnov Test
    kstest_stat, kstest_p_value = kstest(fares, 'norm', args=(fares.mean(), fares.std()))
    print(f"Kolmogorov smirnov test: Statistic={kstest_stat:.4f}, p-value={kstest_p_value:.4f}")

    if shapiro_p_value < 0.05 or kstest_p_value < 0.05:
        print("The fare distribution deviates from a normal distribution (p < 0.05).")
    else:
        print("The fare distribution does not deviate from a normal distribution (p >= 0.05).")

# Function to calculate and print survival rate by embarkation point
def survival_rate_by_embarkation(data):
    survival_by_embarkation = data.groupby('Embarkation Country')['Survived'].mean()
    print("Survival Rate by Embarkation Point:")
    print(survival_by_embarkation)
    return survival_by_embarkation

# Function to plot survival rate by embarkation point
def plot_survival_rate_by_embarkation(data, output_folder):
    plt.figure(figsize=(8, 5))
    sns.barplot(data=data, x='Embarkation Country', y='Survived', estimator=lambda x: x.mean(), ci=None)
    plt.title('Survival Rate by Embarkation Point')
    plt.xlabel('Embarkation Point')
    plt.ylabel('Survival Rate')
    plt.savefig(f"{output_folder}/survival_rate_by_embarkation_point.png")
    plt.close()

# Function to plot stacked bar chart of survival counts by embarkation point
def plot_survival_counts_by_embarkation(data, output_folder):
    survival_counts_by_embarkation = data.groupby(['Embarkation Country', 'Survived']).size().unstack()
    survival_counts_by_embarkation.plot(kind='bar', stacked=True, color=['#ff6666', '#66b3ff'], figsize=(8, 6))
    plt.title('Survival Counts by Embarkation Point')
    plt.xlabel('Embarkation Point')
    plt.ylabel('Passenger Count')
    plt.legend(['Did Not Survive', 'Survived'])
    plt.savefig(f"{output_folder}/survival_counts_by_embarkation_point.png")
    plt.close()

# Function to plot survival rate by embarkation point and ticket class
def plot_survival_by_embarkation_and_class(data, output_folder):
    survival_by_embarkation_class = data.groupby(['Embarkation Country', 'Ticket Class'])['Survived'].mean().unstack()
    plt.figure(figsize=(10, 6))
    survival_by_embarkation_class.plot(kind='bar', stacked=False, figsize=(10, 6))
    plt.title('Survival Rate by Embarkation Point and Ticket Class')
    plt.xlabel('Embarkation Point')
    plt.ylabel('Survival Rate')
    plt.legend(title='Ticket Class')
    plt.savefig(f"{output_folder}/survival_rate_by_embarkationpoint_ticketclass.png")
    plt.close()

def analyze_women_children_by_embarkation(data, output_folder):
    # Define children as passengers under 18
    data['IsChild'] = data['Age'] < 18

    # Filter for women and children
    women_children = data[(data['Gender'] == 'female') | (data['IsChild'])]
    
    embarkation_count = women_children.groupby(['Embarkation Country', 'Gender', 'IsChild']).size().unstack(fill_value=0)

    # Filter embarkation points
    women_children_C_Q_S = women_children[women_children['Embarkation Country'].isin(['C', 'Q', 'S'])]
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=women_children_C_Q_S, x='Embarkation Country', hue='Gender', palette='pastel')
    plt.title('Count of Women and Children by Embarkation Point')
    plt.xlabel('Embarkation Point')
    plt.ylabel('Count')
    plt.legend(title='Demographic')
    plt.savefig(f"{output_folder}/women_children_by_embarkation.png")
    plt.close()
    
    # Ticket class distribution for women and children by embarkation point
    ticket_class_distribution = women_children.groupby(['Embarkation Country', 'Gender', 'IsChild', 'Ticket Class']).size().unstack(fill_value=0)
    print("\nTicket class distribution for women and children by embarkation point:\n", ticket_class_distribution)

def analyze_distribution_by_embarkation(data, output_folder):
    embarkation_counts = data['Embarkation Country'].value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=embarkation_counts.index, y=embarkation_counts.values, palette="pastel")
    plt.title('Passenger Count by Embarkation Point')
    plt.xlabel('Embarkation Point')
    plt.ylabel('Passenger Count')
    plt.savefig(f"{output_folder}/distribution_by_embarkation.png")
    plt.close()

# Analyze influence of embarkation on survival
def analyze_embarkation_survival(data, output_folder):
    # Calculate survival rate by embarkation
    survival_rate_by_embarkation(data)
    
    # Plot survival rate by embarkation point
    plot_survival_rate_by_embarkation(data, output_folder)
    
    # Plot survival counts by embarkation point
    plot_survival_counts_by_embarkation(data, output_folder)
    
    # Plot survival rate by embarkation point and ticket class
    plot_survival_by_embarkation_and_class(data, output_folder)

    analyze_women_children_by_embarkation(data , output_folder)

    analyze_distribution_by_embarkation(data, output_folder)

def analyze_sibling_spouse(data, output_folder):
    # Calculate survival rate by number of siblings/spouse
    sibling_spouse_survival = data.groupby('NumSiblingSpouse')['Survived'].mean()
    sibling_spouse_counts = data.groupby(['NumSiblingSpouse', 'Survived']).size().unstack(fill_value=0)

    # Plot survival rate
    plt.figure(figsize=(8, 5))
    sns.barplot(x=sibling_spouse_survival.index, y=sibling_spouse_survival.values, palette="pastel")
    plt.title('Survival Rate by Number of Siblings/Spouse')
    plt.xlabel('Number of Siblings/Spouse')
    plt.ylabel('Survival Rate')
    plt.savefig(f"{output_folder}/sibling_spouse_survival_rate.png")
    plt.close()

    # Plot counts for survival and non-survival by number of siblings/spouse
    sibling_spouse_counts.plot(kind='bar', stacked=True, color=['#ff6666', '#66b3ff'], figsize=(8, 6))
    plt.title('Survival Counts by Number of Siblings/Spouse')
    plt.xlabel('Number of Siblings/Spouse')
    plt.ylabel('Passenger Count')
    plt.legend(['Did Not Survive', 'Survived'])
    plt.savefig(f"{output_folder}/sibling_spouse_survival_counts.png")
    plt.close()

    data['IsChild'] = data['Age'] < 18
    # Plot survival rate by number of siblings and gender
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='NumSiblingSpouse', y='Survived', hue='Gender', ci=None, palette='pastel')
    plt.title('Survival Rate by Number of Siblings/Spouse and Gender')
    plt.xlabel('Number of Siblings/Spouse')
    plt.ylabel('Survival Rate')
    plt.legend(title='Gender')
    plt.savefig(f"{output_folder}/survival_by_siblings_gender.png")
    plt.close()

     # Plot survival rate by sibling/spouse count and child status
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='NumSiblingSpouse', y='Survived', hue='IsChild', ci=None, palette='muted')
    plt.title('Survival Rate by Number of Siblings/Spouse and Child Status')
    plt.xlabel('Number of Siblings/Spouse')
    plt.ylabel('Survival Rate')
    plt.legend(title='Child Status (Under 18)')
    plt.savefig(f"{output_folder}/survival_by_siblings_child.png")
    plt.close()

    # Plot survival rate by sibling/spouse count and ticket class
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='NumSiblingSpouse', y='Survived', hue='Ticket Class', ci=None, palette='Set2')
    plt.title('Survival Rate by Number of Siblings/Spouse and Ticket Class')
    plt.xlabel('Number of Siblings/Spouse')
    plt.ylabel('Survival Rate')
    plt.legend(title='Ticket Class')
    plt.savefig(f"{output_folder}/survival_by_siblings_ticket_class.png")
    plt.close()

def analyze_survival_by_parent_children(data, output_folder):
    # Calculate survival count by number of parent/child
    survival_counts = data.groupby(['NumParentChild', 'Survived']).size().unstack(fill_value=0)
    survival_counts.plot(kind='bar', stacked=True, color=['#ff6666', '#66b3ff'], figsize=(10, 6))
    plt.title('Survival Counts by Number of Parent/Child')
    plt.xlabel('Number of Parent/Child')
    plt.ylabel('Passenger Count')
    plt.legend(['Did Not Survive', 'Survived'])
    plt.savefig(f"{output_folder}/parent_child_survival_counts.png")
    plt.close()

    # Plot survival rate by number of parent/child
    survival_rate = data.groupby('NumParentChild')['Survived'].mean()
    survival_rate.plot(kind='bar', color='#66b3ff', figsize=(10, 6))
    plt.title('Survival Rate by Number of Parent/Child')
    plt.xlabel('Number of Parent/Child')
    plt.ylabel('Survival Rate')
    plt.savefig(f"{output_folder}/parent_child_survival_rate.png")
    plt.close()

    # Plot survival rate by number of parent/child and ticket class
    survival_by_class = data.groupby(['NumParentChild', 'Ticket Class'])['Survived'].mean().unstack()
    survival_by_class.plot(kind='bar', stacked=False, figsize=(10, 6), color=['#4daf4a', '#ff7f00', '#377eb8'])
    plt.title('Survival Rate by Number of Parent/Child and Ticket Class')
    plt.xlabel('Number of Parent/Child')
    plt.ylabel('Survival Rate')
    plt.legend(title='Ticket Class')
    plt.savefig(f"{output_folder}/parent_child_survival_by_ticket_class.png")
    plt.close()

    # Plot survival rate by number of parent/child and child status which refer to less than 18yo
    data['IsChild'] = data['Age'] < 18
    survival_by_child_status = data.groupby(['NumParentChild', 'IsChild'])['Survived'].mean().unstack()
    survival_by_child_status.plot(kind='bar', stacked=False, figsize=(10, 6), color=['#66b3ff', '#ff6666'])
    plt.title('Survival Rate by Number of Parent/Child and Child Status')
    plt.xlabel('Number of Parent/Child')
    plt.ylabel('Survival Rate')
    plt.legend(['Adult', 'Child'], title='Age Group')
    plt.savefig(f"{output_folder}/parent_child_survival_by_child_status.png")
    plt.close()

    # Plot survival rate by number of parents/children and gender.
    survival_by_gender = data.groupby(['NumParentChild', 'Gender'])['Survived'].mean().unstack()
    survival_by_gender.plot(kind='bar', stacked=False, figsize=(10, 6), color=['#66b3ff', '#ff6666'])
    plt.title('Survival Rate by Number of Parent/Child and Gender')
    plt.xlabel('Number of Parent/Child')
    plt.ylabel('Survival Rate')
    plt.legend(title='Gender')
    plt.savefig(f"{output_folder}/parent_child_survival_by_gender.png")
    plt.close()

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

# Main EDA function
def eda_q3(data_path, output_folder):
    # Load data
    data = pd.read_csv(data_path)

    # Generate total survivors and deaths pie chart
    plot_survival_by_group(data, group_by=None, output_folder=output_folder, chart_type='pie')

    # Test hypothesis 1: survival by gender (bar chart)
    plot_survival_by_group(data, group_by='Gender', output_folder=output_folder, chart_type='bar')

    
    plot_survival_by_group(data, group_by='Ticket Class', output_folder=output_folder, chart_type='bar')

    # Print survival and death numbers by ticket class and gender
    print_survival_death_by_class_and_gender(data)
    
    # Analyze survival rate by gender and class
    plot_survival_rate_by_gender_and_class(data, output_folder)

    # Analyze survival by age
    analyze_survival_by_age(data, output_folder)

    analyze_survival_by_ticket_price(data, output_folder)

    analyze_fare_distribution(data, output_folder)

    analyze_embarkation_survival(data, output_folder)

    analyze_sibling_spouse(data, output_folder)

    analyze_survival_by_parent_children(data, output_folder)

    print(f"EDA completed. Results saved in {output_folder}.")

# Main function
def main(config_file="config.txt"):
    # Open config file for review
    if not open_config_file(config_file):
        print("Exiting due to missing or inaccessible config file.")
        return

    # Load paths and suffix
    paths = load_eda_paths_and_suffix(config_file)
    input_folder = paths["input_folder"]
    output_folder = paths["output_folder"]
    eda_file_suffix = paths["eda_file_suffix"]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List available files and prompt for selection
    files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
    if not files:
        print("No files found in the cleaned data folder.")
        return

    print("Available files in cleaned data folder:")
    for i, file in enumerate(files, 1):
        print(f"{i}. {file}")

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

    input_path = os.path.join(input_folder, selected_file)

    # Define output filename with suffix
    output_filename = f"eda_{selected_file.split('.')[0]}{eda_file_suffix}.csv"
    output_path = os.path.join(output_folder, output_filename)

    # Run EDA and save results to output folder
    eda_q3(input_path, output_folder)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
#     parser.add_argument("--input_folder", type=str, default="cleaned_data", help="Path to the cleaned data folder.")
#     parser.add_argument("--output_folder", type=str, default="eda_outputs", help="Folder to save the EDA outputs.")
    
#     args = parser.parse_args()
    
#     # List available files in cleaned_data
#     files = os.listdir(args.input_folder)
#     print("Available files in cleaned_data:")
#     for idx, file in enumerate(files):
#         print(f"{idx + 1}: {file}")

#     # Prompt user for file selection
#     file_choice = int(input("Select the file number to perform EDA on: ")) - 1
#     selected_file = files[file_choice]
#     data_path = os.path.join(args.input_folder, selected_file)

#     # Run EDA on the selected file and save results to output folder
#     eda_q3(data_path, args.output_folder)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
#     parser.add_argument("--input_folder", type=str, default="../OTH_DATA/cleaned_data", help="Path to the cleaned data folder.")
#     parser.add_argument("--output_folder", type=str, default="../EDA_DATA", help="Folder to save the EDA outputs.")
    
#     args = parser.parse_args()
    
#     # List available files in cleaned_data
#     files = os.listdir(args.input_folder)
#     print("Available files in cleaned_data:")
#     for idx, file in enumerate(files):
#         print(f"{idx + 1}: {file}")

#     # Prompt user for file selection
#     file_choice = int(input("Select the file number to perform EDA on: ")) - 1
#     selected_file = files[file_choice]
#     data_path = os.path.join(args.input_folder, selected_file)

#     # Run EDA on the selected file and save results to output folder
#     eda_q3(data_path, args.output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on cleaned data.")
    parser.add_argument('--config_file', type=str, default="config.txt", help="Path to the configuration file.")
    args = parser.parse_args()
    main(args.config_file)