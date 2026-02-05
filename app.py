from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "fitness_data.csv")

data = pd.read_csv(csv_path)


le = LabelEncoder()
data["FitnessLevel"] = le.fit_transform(data["FitnessLevel"])

X = data[["Age", "BMI", "Steps", "HeartRate"]]
y = data["FitnessLevel"]

model = DecisionTreeClassifier()
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    suggestions = []

    if request.method == "POST":
        name = request.form["name"]
        age = int(request.form["age"])
        height_cm = float(request.form["height"])
        weight = float(request.form["weight"])
        steps = int(request.form["steps"])
        heart_rate = int(request.form["heart_rate"])

        # Convert cm to meters
        height_m = height_cm / 100

        # BMI calculation
        bmi = weight / (height_m ** 2)

        # AI Prediction
        pred = model.predict([[age, bmi, steps, heart_rate]])
        fitness_level = le.inverse_transform(pred)[0]

        # AI-based suggestions
        if fitness_level == "Low":
            suggestions = [
                "Start with light physical activity",
                "Maintain a balanced diet",
                "Improve sleep and hydration"
            ]
        elif fitness_level == "Medium":
            suggestions = [
                "Increase activity gradually",
                "Include cardio and strength training",
                "Stay consistent"
            ]
        else:
            suggestions = [
                "Maintain your routine",
                "Focus on recovery",
                "Continue healthy habits"
            ]

        result = {
            "name": name,
            "bmi": round(bmi, 2),
            "fitness": fitness_level
        }

    return render_template("index.html", result=result, suggestions=suggestions)
    
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)


