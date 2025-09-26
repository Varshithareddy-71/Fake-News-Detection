from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load your trained model and vectorizer
vectorizer = joblib.load("C:/Users/Varshitha Reddy/Downloads/Fake_news/Fake_news/vectorizer.pkl")
model = joblib.load("C:/Users/Varshitha Reddy/Downloads/Fake_news/Fake_news/model.pkl")
DT = joblib.load("C:/Users/Varshitha Reddy/Downloads/Fake_news/Fake_news/DT.pkl")
RFC = joblib.load("C:/Users/Varshitha Reddy/Downloads/Fake_news/Fake_news/RFC.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    news = request.form["news"]
    vect_news = vectorizer.transform([news])
    prediction1 = model.predict(vect_news)[0]
    prediction2 = DT.predict(vect_news)[0]
    prediction3 = RFC.predict(vect_news)[0]
    result1 = "Real News 🟢" if prediction1 == 1 else "Fake News 🔴"
    result2 = "Real News 🟢" if prediction2 == 1 else "Fake News 🔴"
    result3 = "Real News 🟢" if prediction3 == 1 else "Fake News 🔴"
    return render_template("index.html", prediction1=result1, prediction2=result2, prediction3=result3, user_input=news)

if __name__ == "__main__":
    app.run(debug=True)
