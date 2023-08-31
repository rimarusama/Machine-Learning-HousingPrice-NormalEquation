# Machine-Learning-HousingPrice-NormalEquation

This repository contains a simple implementation of linear regression using the normal equation method, along with a manual train-test split with a reproducible random seed. The code is written in Python and utilizes numpy for matrix operations and pandas for data manipulation.

## Dataset

The dataset used in this example is named `housing.csv`. It should be placed in the same directory as the code files. The dataset contains features of houses and their corresponding prices.

## Usage

1. Clone the repository:
   https://github.com/sudammajhi/Machine-Learning-HousingPrice-NormalEquation.git
   cd Machine-Learning-HousingPrice-NormalEquation
   
3. Place your `housing.csv` dataset file in the repository directory.

4. Run the main script:
   python linear_regression.py

5.The script will perform the following steps:
- Read the dataset and split it into training and test sets using a reproducible random seed.
- Calculate the linear regression coefficients using the normal equation method on the training set.
- Make predictions on both the training and test sets.
- Calculate and display the R-squared scores for both sets.

## Requirements

- Python 3.x
- numpy
- pandas
- matplotlib

Install the required libraries using:
  import numpy pandas matplotlib
