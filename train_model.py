from pathlib import Path
import pandas as pd
import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv(BASE_DIR / "mail_data.csv")

# Handle missing values
df = df.where(pd.notnull(df), '')

# Encode labels
df["Category"] = df["Category"].map({"ham": 0, "spam": 1})

X = df["Message"]
y = df["Category"]

# Vectorize text
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model + vectorizer
with open(MODELS_DIR / "model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open(MODELS_DIR / "vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("✅ Model and vectorizer saved successfully!")
