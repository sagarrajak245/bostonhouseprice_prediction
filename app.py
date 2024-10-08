# app.py
from flask import Flask, request, jsonify, render_template
import joblib     

app = Flask(__name__)

# Load your model here
model = joblib.load('models/boston_house_price_prediction.pkl')

@app.route("/") 
def home_page():
    return render_template('index.html') 

@app.route("/predict", methods=['POST']) 
def predict():
    data = request.json['data']
    # Perform prediction using your model
    prediction = model.predict([data])  # Adjust based on your model input
    prediction = "Sample Prediction"  # Replace with actual prediction
    return jsonify({'prediction': prediction})

if __name__ == '__main__': 
    app.run(debug=True)