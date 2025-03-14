from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from model import price_predict, train_model
from data_preprocessing import preprocess_data

app = Flask(__name__)

# Load and preprocess the dataset
DATA_PATH = "House.csv"
house_data = pd.read_csv(DATA_PATH)
house_data_preprocessed, X, y = preprocess_data(house_data)
model = train_model(X, y)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    location = data['location']
    sqft = float(data['sqft'])
    bath = int(data['bath'])
    bhk = int(data['bhk'])
    price = price_predict(location, sqft, bath, bhk, model, X)
    return jsonify({'price': price})

if __name__ == '__main__':
    app.run(debug=True)
