from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model and scaler
with open("fish_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        data = [float(request.form[key]) for key in ["Length1", "Length2", "Length3", "Height", "Width"]]
        
        # Preprocess input
        data_scaled = scaler.transform([data])
        prediction = model.predict(data_scaled)[0]
        
        return jsonify({"Predicted Weight": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
