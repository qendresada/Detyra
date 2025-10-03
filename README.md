  # AG News Text Classification – AI Internship Challenge

This repository contains my submission for the LinkPlus AI Internship Challenge.  
The project trains a simple text classification model on the AG News dataset with four categories: World, Sports, Business, and Sci/Tech.

## Dataset
AG News dataset (train/test CSVs):  
https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset?resource=download

Download `AGNEWS_TRAIN.csv` and `AGNEWS_TEST.csv` and update the file paths in the code if needed.

## Installation
```
pip install pandas scikit-learn
```
## How to run
Option 1 – Python script:
```
python detyra.py
```

Option 2 – Jupyter notebook:
```
jupyter notebook detyra.ipynb
```

## Output
When executed, the script will:
- Print the number of samples per category
- Print the most frequent words in the dataset
- Train and evaluate a Logistic Regression model with TF-IDF features
- Show the classification report with precision, recall and F1-score
- Print train and test accuracy
- Provide example predictions for new sentences
- Visualize:
  - Category distribution (bar chart)
  - Precision/Recall/F1-scores per category (bar chart)
