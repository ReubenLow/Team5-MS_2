# Jupyter Notebooks for Python 3.11

This directory contains Jupyter notebooks compatible with Python 3.11.

## Notebooks are located in: 
	- ML_PIPELINE_DEMO that runs data cleaning -> model training -> perform prediction.
	- EDA_PIPELINE_DEMO that runs data cleaning -> perform exploratory data analysis on data set.
	- COMBINED_PIPELINE_DEMO that runs the entire pipeline.

- `clean_data.ipynb`: Notebook for data analysis.
- `perform_eda.ipynb`: Notebook for data analysis.
- `model_training.ipynb`: Notebook for training machine learning models.
- `perform_prediction.ipynb`: Notebook for evaluating trained models by performing predictions on test datasets.

## Instructions

1. **Set Up Environment**:
   - Ensure Python 3.11 is installed on your system.
   - Create a virtual environment:
     ```
     python -m venv venv
     ```
   - Activate the virtual environment:
     - On Windows:
       ```
       source myenv/Scripts/activate
       ```
   - Install required packages:
     ```
     pip install -r requirements.txt
     ```

2. **Launch Jupyter Notebook**:
   - Start the Jupyter Notebook server:
     In either the following folders: ML_PIPELINE_DEMO, EDA_PIPELINE_DEMO, COMBINED_PIPELINE_DEMO
     ```
     jupyter notebook
     ```
   - Open the desired notebook (e.g., `clean_data.ipynb`) in your browser.

3. **Run Notebooks**:
   - Execute the cells sequentially.
   - Ensure all dependencies are installed and paths to data files are correct.

## Notes

- Modify `requirements.txt` to add or remove packages as needed.
- Ensure data files are in the correct directories as referenced in the notebooks.
