from pathlib import Path
import subprocess
import sys
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def ensure_model_files():
    model_path = MODELS_DIR / "model.pkl"
    vectorizer_path = MODELS_DIR / "vectorizer.pkl"

    if model_path.exists() and vectorizer_path.exists():
        return

    MODELS_DIR.mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(BASE_DIR / "train_model.py")], check=True)


def load_pickle(filename: str):
    file_path = BASE_DIR.joinpath(filename)
    with open(str(file_path), "rb") as file:
        return pickle.load(file)


# Load model (example)
ensure_model_files()
model = load_pickle("models/model.pkl")
vectorizer = load_pickle("models/vectorizer.pkl")

# Welcome page
@app.route("/")
def index():
    return render_template("index.html")


# Main app page
@app.route("/home")
def home():
    return render_template("home.html")


# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    result = "Spam" if prediction[0] == 1 else "Not Spam"
    return render_template("home.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)
