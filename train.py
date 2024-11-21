import json
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Load data
with open("app/intents.json", "r") as file:
    intents = json.load(file)

# Preprocessing
X_train = []
y_train = []
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X_train.append(pattern)
        y_train.append(intent["tag"])

# Vectorize the patterns
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)

# Save the model and vectorizer
with open("models/intent_model.pkl", "wb") as model_file:
    pickle.dump((model, vectorizer), model_file)

print("Model trained and saved successfully!")
