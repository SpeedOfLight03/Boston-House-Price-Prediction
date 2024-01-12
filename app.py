import pickle
import json
from flask import Flask, jsonify, render_template, request, url_for
import numpy as np
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict_api", methods=["POST"])
def predict_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = model.predict(new_data)
    print(output[0])
    return jsonify(output[0])


@app.route("/predict", methods=["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_inp = scaler.transform(np.array(data).reshape(1, -1))
    print(final_inp)
    output = model.predict(final_inp)
    return render_template(
        "index.html", prediction_text=f"The price of the house is {output[0]}"
    )


if __name__ == "__main__":
    app.run(debug=True)
