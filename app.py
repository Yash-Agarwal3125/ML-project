from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd

app = Flask(__name__)

# Load custom weights and bias
model = pickle.load(open("model.pkl", "rb"))
fin_w = model['weights']
fin_b = model['bias']

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')     #basic code to render the index.html file from templates folder
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])  # predict using the model.pkl that we got from final_flight.py
def predict():
    try:
        # Get form data
        source_city = int(request.form['source_city'])
        destination_city = int(request.form['destination_city'])
        flight_class = int(request.form['class'])
        duration = float(request.form['duration'])
        days_left = int(request.form['days_left'])
        departure_time = int(request.form['departure_time'])

        # Input array in correct order
        x_input = np.array([source_city, destination_city, flight_class, duration, days_left, departure_time]).reshape(1, -1)

        # Scale it
        x_scaled = scaler.transform(x_input)

        # Predict
        predicted_price = np.dot(fin_w, x_scaled[0]) + fin_b

        return render_template('index.html', prediction_text=f"Predicted Flight Price: ₹{predicted_price:.2f}")
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)  #if any error it will sow on webpage 
