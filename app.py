
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("churn_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        tenure = int(request.form["tenure"])
        monthly_charges = float(request.form["MonthlyCharges"])
        total_charges = float(request.form["TotalCharges"])

        contract = request.form["Contract"]
        internet_service = request.form["InternetService"]
        payment_method = request.form["PaymentMethod"]

        if not 0 <= tenure <= 100:
            raise ValueError("Please enter a valid tenure.")

        if monthly_charges < 0 or total_charges < 0:
            raise ValueError("Charges cannot be negative.")

        customer_data = pd.DataFrame([{
            "tenure": tenure,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Contract": contract,
            "InternetService": internet_service,
            "PaymentMethod": payment_method
        }])

        prediction = model.predict(customer_data)[0]

        probabilities = model.predict_proba(customer_data)[0]
        churn_index = list(model.classes_).index("Yes")
        churn_probability = probabilities[churn_index] * 100

        if prediction == "Yes":
            result = "Customer is likely to Churn"
        else:
            result = "Customer is likely to Stay"

        return render_template(
            "index.html",
            prediction=result,
            churn_probability=round(churn_probability, 2),
            prediction_class="danger" if prediction == "Yes" else "success"
        )

    except (ValueError, KeyError) as error:
        return render_template(
            "index.html",
            error=str(error)
        )


if __name__ == "__main__":
    app.run(debug=True)
