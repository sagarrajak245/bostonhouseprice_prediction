# Boston House Price Prediction
sagar..here this is my first ml project which i done by myself writing each and every line from scratch and solving self doubts if this is happening then why if this then why not other and many more situation like this..also hypertuning is great concept i will try my best to utilize that but it didnnt worked here too much..i ustilized flask ,html/css/js to make web page and dockerize my web app.....thanks if u are here ..not very fancy project i know  but my first one and i loved while exploring everything about linear regression so yes it is like that....
This project predicts house prices in Boston using a machine learning model. The model is built using Python, and the project includes Exploratory Data Analysis (EDA), model training with Scikit-learn, a web interface using Flask, and Docker for containerization.

## Project Structure


- `app/`: Contains the Flask application code.
  - `static/`: Stores static files (CSS, JS, images).
  - `templates/`: Stores HTML templates for the web interface.
  - `model.pkl`: Trained machine learning model.
  - `app.py`: Flask application to handle requests and serve predictions.
  - `requirements.txt`: Lists all dependencies needed for the app.
  
- `data/`: Stores the dataset (Boston housing data in CSV format).
- `Dockerfile`: Defines the container setup for Docker.
- `notebooks/`: Jupyter notebooks used for Exploratory Data Analysis (EDA).

## Tech Stack

- **Frontend**: HTML, CSS, JavaScript (via Flask)
- **Backend**: Python (Flask)
- **Machine Learning**: Scikit-learn
- **Containerization**: Docker

## Features

- Perform exploratory data analysis on the Boston housing dataset.
- Train a machine learning model to predict house prices based on various features.
- Provide a web interface for users to input house details and receive predicted prices.
- Containerized the Flask application using Docker for easy deployment.

## Prerequisites

Before running this project, ensure that you have the following installed:

- Python 3.x
- Flask
- Docker
- Jupyter Notebook (for running EDA)

## Getting Started

### 1. Clone the repository:

```bash
git clone https://github.com/sagarrajak245/boston-house-price-prediction.git
cd boston-house-price-prediction
pip install -r requirements.txt
python app.py
docker build -t boston-house-price-app .
docker run -p 5000:5000 boston-house-price-app

