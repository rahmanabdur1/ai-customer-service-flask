import pickle
import random
import json
# Load the model and vectorizer
with open("models/intent_model.pkl", "rb") as model_file:
    model, vectorizer = pickle.load(model_file)

def get_response(user_input):
    user_input_vectorized = vectorizer.transform([user_input])
    prediction = model.predict(user_input_vectorized)[0]

    # Load intents
    with open("app/intents.json", "r") as file:
        intents = json.load(file)

    for intent in intents["intents"]:
        if intent["tag"] == prediction:
            return random.choice(intent["responses"])

    return "I'm sorry, I didn't understand that."
