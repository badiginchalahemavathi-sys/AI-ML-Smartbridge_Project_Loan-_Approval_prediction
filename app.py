from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model/model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    gender = int(request.form["gender"])
    married = int(request.form["married"])
    dependents = int(request.form["dependents"])
    education = int(request.form["education"])
    self_employed = int(request.form["self_employed"])
    applicant_income = float(request.form["income"])
    coapplicant_income = float(request.form["co_income"])
    loan_amount = float(request.form["loan"])
    loan_term = float(request.form["loan_term"])
    credit_history = int(request.form["credit"])
    property_area = int(request.form["property_area"])

    features = np.array([[gender,
                          married,
                          dependents,
                          education,
                          self_employed,
                          applicant_income,
                          coapplicant_income,
                          loan_amount,
                          loan_term,
                          credit_history,
                          property_area]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "Loan Approved"
    else:
        result = "Loan Rejected"

    return render_template("result.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)