from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

# Load custom weights and bias
model = pickle.load(open("model.pkl", "rb"))
fin_w = model['weights']
fin_b = model['bias']

# Load scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from fetch() request
        data = request.get_json()

        # Extract and convert input fields
        source_city = int(data['source_city'])
        destination_city = int(data['destination_city'])
        flight_class = int(data['class'])
        duration = float(data['duration'])
        days_left = int(data['days_left'])
        departure_time = int(data['departure_time'])

        # Prepare and scale input
        x_input = np.array([source_city, destination_city, flight_class, duration, days_left, departure_time]).reshape(1, -1)
        x_scaled = scaler.transform(x_input)

        # Predict using weights and bias
        predicted_price = np.dot(fin_w, x_scaled[0]) + fin_b

        return jsonify({'predicted_price': round(predicted_price, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
