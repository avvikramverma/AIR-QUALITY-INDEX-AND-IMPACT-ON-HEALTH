from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.getcwd(), 'rm.pkl')
with open(model_path, 'rb') as f:
    loaded_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extracting input data from the form
        aqi = float(request.form['aqi'])
        pm10 = float(request.form['pm10'])
        pm2_5 = float(request.form['pm2_5'])
        no2 = float(request.form['no2'])
        so2 = float(request.form['so2'])
        o3 = float(request.form['o3'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        windspeed = float(request.form['windspeed'])
        cardiovascular = float(request.form['cardiovascular'])
        respiratory = float(request.form['respiratory'])
        hospital_admissions = float(request.form['hospital_admissions'])

        # Forming the feature vector for the prediction
        features = np.array([[aqi, pm10, pm2_5, no2, so2, o3, temperature, humidity, windspeed, cardiovascular, respiratory, hospital_admissions]])

        # Making predictions
        predictions = loaded_model.predict(features)

        # If predictions return a single target (1D array)
        # Adjust based on the output type of your model
        if len(predictions.shape) == 1:  # Single output case
            health_impact_class = int(predictions[0])  # For classification
            health_impact_score = predictions[0]        # For regression (if applicable)
        else:  # Multi-output case
            health_impact_class = int(predictions[0][0])  # First target (classification)
            health_impact_score = predictions[0][1]       # Second target (score)

        # Render the results on the webpage
        result = {
            "Health Impact Class": health_impact_class,
            "Health Impact Score": health_impact_score
        }

        return render_template('index.html', prediction=result)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
