# Best Mobile Tariff Predictor

A machine learning-based system for predicting the most suitable mobile tariff plans for customers based on their usage patterns.

## Project Overview

This project analyzes customer mobile usage data and predicts the most cost-effective tariff plans using Support Vector Machine (SVM) model. It processes historical customer usage data including data consumption, minutes, and SMS usage to recommend optimal mobile plans.

## Features

- Customer usage data analysis
- SVM model for tariff prediction
- Support for both historical and new customer data
- Cost optimization calculations
- Batch processing capabilities

## Project Structure 
task_mobile/
├── output/                                            
│   └── plan_comparison_20241119_140429.csv  
├── src/
│ └── insights/
│ ├── CSV_files/
│ │ ├── Customer_Usage_Last_12_Months.csv
│ │ ├── Customer_Usage_Last_12_Months_new.csv
│ │ └── Mobile_Plans_Test_Data.csv
│ ├── data_analysis/ 
│ │ └── data_manipulation.ipynb
│ └── task_analysis/
│ ├── AI_test_new.py 
│ ├── AI_test_old.py 
│ ├── mobile_analysis.py 
│ ├── model_predictor.py 
│ └── model_train.ipynb 
├── .devcontainer/ 
│ └── Dockerfile
├── .gitignore
├── README.md
└── poetry.lock 


## Requirements

- Docker
- Python 3.10.14
- Poetry

## Installation & Setup

1. Clone the repository:

```bash
git clone https://github.

2. Build and run the Docker container:


## Contact

- **Developer**: Amer Alnajdawi
- **Email**: amernajdawi8@gmail.com
- **GitHub**: [@amernajdawi](https://github.com/amernajdawi)