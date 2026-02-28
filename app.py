from flask import Flask, render_template, request, jsonify
import json
import random
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# SIH Knowledge Base
data = {
    "questions": [
        "What is SIH?",
        "How to prepare for SIH?",
        "What topics are good for SIH?",
        "How to select a problem statement?",
        "How to build a strong SIH team?",
        "What technologies can be used in SIH?",
        "How to win SIH?",
        "What is required in SIH prototype?",
        "What are key points that we have to tell during presentation?"
    ],
    "answers": [
        "Smart India Hackathon is a nationwide innovation competition for students to solve real-world problems.",
        "Start by understanding the problem statement, choose the right tech stack, and build a working prototype.",
        "AI, Machine Learning, Web Development, IoT, Cybersecurity, Blockchain are good topics.",
        "Choose a problem that is practical, solvable, and matches your team's skills.",
        "A strong SIH team should have developers, designers, and presenters.",
        "You can use Python, Flask, React, AI/ML, IoT, Cloud computing technologies.",
        "Focus on innovation, working prototype, clear presentation, and teamwork.",
        "A working model, PPT presentation, documentation, and demo video are required.",
        "Firstly EXplain what your exact problem is and what are technologies you have used in it.Secondly explain how your project is different from the exisiting one."
    ]
}

# Combine Q&A for ML training
corpus = data["questions"] + data["answers"]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

def chatbot_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, X)
    index = np.argmax(similarity)
    
    if similarity[0][index] < 0.2:
        return "Sorry, I don't understand that question. Please ask something related to SIH."
    if index < len(data["questions"]):
        return data["answers"][index]
    else:
        return data["answers"][index - len(data["questions"])]
    
    return data["answers"][index]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.form["msg"]
    return chatbot_response(user_input)

if __name__ == "__main__":
    app.run(debug=True)