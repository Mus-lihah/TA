from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained Random Forest model
rf_model = joblib.load("rf.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        features = pd.DataFrame([features], columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'])

        prediction = rf_model.predict(features)

        # Map the prediction result to a meaningful label
        result = "Good" if prediction[0] == 1 else "Bad"

        return render_template('index.html', prediction_text='Predicted Wine Quality: {}'.format(result))

if __name__ == '__main__':
    app.run(debug=True)
