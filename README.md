# Anime Recommender System

![Movie_Recommendations](resources/imgs/Image_header.png)

## Overview

Welcome to the Anime Recommender Streamlit System project! This project aims to build a collaborative and content-based recommender system based on Streamlit for a collection of anime titles. The system predicts how a user will rate an anime title they have not yet viewed, based on their historical preferences.

## Table of Contents

- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Streamlit Application](#streamlit-application)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This project involves several key components:
- Data loading and preprocessing
- Model training and evaluation
- Building a recommender system using collaborative filtering and content-based filtering
- Developing a Streamlit web application for user interaction
- Deployment and final presentation

## Environment Setup

To set up the environment for this project, follow these steps:

### Step 1: Install Anaconda

Download and install Anaconda from [here](https://www.anaconda.com/products/distribution).

### Step 2: Create a New Conda Environment

Create a new conda environment using the `requirements.txt` file.

```bash

# Create a new environment (e.g., named 'anime_recommender')
conda create --name anime_recommender python=3.8

# Activate the new environment
conda activate anime_recommended

### Step 3: Install Required Packages
Install the packages listed in requirements.txt.
```
### Step 3: Install Required Packages

Install the packages listed in requirements.txt.

```bash
pip install -r requirements.txt
```
### Step 4: Verify Installation
Ensure all packages are installed correctly by running a script or checking the versions of the packages.

```bash
# Example: Check package versions
python -c "import numpy; print(numpy.__version__)"
```
## Project Structure
The project structure is as follows

```css
├── data
│   ├── raw
│   ├── processed
├── notebooks
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── model_evaluation.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── model_evaluation.py
├── streamlit_app
│   ├── app.py
│   ├── pages
│       ├── homepage.py
│       ├── recommendations.py
├── requirements.txt
├── README.md
└── LICENSE

```

## Usage
Data Preprocessing
To preprocess the data, run the data_preprocessing.ipynb notebook or execute the corresponding script:

```bash
python src/data_preprocessing.py
```
## Model Training and Evaluation
To train and evaluate the models, use the model_training.ipynb and model_evaluation.ipynb notebooks or execute the corresponding scripts:

```bash
python src/model_training.py
python src/model_evaluation.py
```
## Streamlit Application
To run the Streamlit application, execute the following command:

``` bash
streamlit run streamlit_app/app.py
```
Important Links
Facilitator GitHub Usernames
Managing Environments
Jupyter Notebook Markdown Cheatsheet
MLflow Guide
Streamlit Guide
Tutorial on Recommender Systems
Configuring Git Large File Storage
Thank you for checking out our project! We hope you find it interesting and useful.


## 7. Team Members<a class="anchor" id="team-members"></a>
| Name                                                                                        |  Email              
|---------------------------------------------------------------------------------------------|--------------------             
| [Silindile Ndlela]()                                                      |  
| [Sandile Jali](a)                                                             | 
| [Queen]()                                                   | 
| [Sinawo]()                                                | 
| [Johannes]()                                         | 
| []()                                                 | 

## Acknowledgments


* 
* 
* 
