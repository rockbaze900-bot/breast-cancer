from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open("../model/model.pkl", "rb"))
scaler = pickle.load(open("../model/scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final = scaler.transform([features])
    prediction = model.predict(final)[0]

    result = "Malignant" if prediction == 0 else "Benign"
    return render_template("index.html", prediction_text=result)

# API endpoint (important for portfolio)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json["features"]
    final = scaler.transform([data])
    prediction = int(model.predict(final)[0])
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
