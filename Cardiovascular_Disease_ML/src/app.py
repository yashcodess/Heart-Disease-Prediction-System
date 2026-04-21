from flask import Flask, render_template, request, redirect, Response
import joblib
import numpy as np
import sqlite3
import pandas as pd 
from datetime import datetime
import re

app = Flask(__name__,
            template_folder="../templates",
            static_folder="../static")

# Load model
model = joblib.load("../model/heart_disease_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

# DB setup
def init_db():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        age REAL,
        sex REAL,
        cp REAL,
        trestbps REAL,
        chol REAL,
        fbs REAL,
        restecg REAL,
        thalach REAL,
        exang REAL,
        oldpeak REAL,
        slope REAL,
        ca REAL,
        thal REAL,
        result TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

init_db()

# HOME
@app.route("/")
def home():
    return render_template("index.html")

# PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    try:
        age = float(request.form["age"])
        sex = float(request.form["sex"])
        cp = float(request.form["cp"])
        trestbps = float(request.form["trestbps"])
        chol = float(request.form["chol"])
        fbs = float(request.form["fbs"])
        restecg = float(request.form["restecg"])
        thalach = float(request.form["thalach"])
        exang = float(request.form["exang"])
        oldpeak = float(request.form["oldpeak"])
        slope = float(request.form["slope"])
        ca = float(request.form["ca"])
        thal = float(request.form["thal"])

        # Validation
        if age < 1 or age > 120:
            return render_template("index.html", prediction_text="Invalid Age")

       
        columns = ["age","sex","cp","trestbps","chol","fbs","restecg",
           "thalach","exang","oldpeak","slope","ca","thal"]

        features = [age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal]

        df = pd.DataFrame([features], columns=columns)

        final_features = scaler.transform(df)

        prediction = model.predict(final_features)
        probability = model.predict_proba(final_features)

        prob = round(probability[0][1] * 100, 2)

        # Risk level
        if prob >= 70:
            level = "High Risk"
        elif prob >= 50:
            level = "Moderate Risk"
        else:
            level = "Low Risk"

        if prediction[0] == 1:
            result = f"Heart Disease Detected ({prob}% - {level})"
        else:
            result = f"No Heart Disease ({prob}% - {level})"

        # Save to DB
        conn = sqlite3.connect("history.db")
        cursor = conn.cursor()

        cursor.execute("""
        INSERT INTO history (
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            result, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal,
            result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return render_template("index.html", prediction_text=result)

    except:
        return render_template("index.html", prediction_text="Invalid Input")

# HISTORY
@app.route("/history")
def history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC LIMIT 6")
    data = cursor.fetchall()
    conn.close()

    total = len(data)

    # ✅ FIXED: Correct risk classification
    high_risk = 0
    safe = 0

    for row in data:
        result = row[14]

        if "High Risk" in result or "Moderate Risk" in result:
            high_risk += 1
        elif "Low Risk" in result:
            safe += 1

    # ✅ Extract risk %
    risk_values = []
    labels = []

    for i, row in enumerate(reversed(data)):
        result = row[14]

        match = re.search(r"(\d+\.?\d*)%", result)
        if match:
            risk_values.append(float(match.group(1)))
        else:
            risk_values.append(0)

        labels.append(f"Case {i+1}")

    return render_template(
        "history.html",
        data=data,
        total=total,
        high_risk=high_risk,
        safe=safe,
        risk_values=risk_values,
        labels=labels
    )

# DELETE
@app.route("/delete_history")
def delete_history():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM history")
    conn.commit()
    conn.close()
    return redirect("/history")

# DOWNLOAD CSV
@app.route("/download_csv")
def download_csv():
    conn = sqlite3.connect("history.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history")
    data = cursor.fetchall()
    conn.close()

    def generate():
        yield "ID,Age,Sex,BP,Chol,Result,Time\n"
        for row in data:
            yield f"{row[0]},{row[1]},{row[2]},{row[4]},{row[5]},{row[14]},{row[15]}\n"

    return Response(generate(),
                    mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=history.csv"})

if __name__ == "__main__":
    print("inside main")
    app.run(debug=True)