from flask import Flask, render_template, request
from load_model import model, predict_sentiment
model.eval()

label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}


app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html")

@app.route("/home", methods=['POST'])
def home():
    text = request.form['text']
    sentiment = predict_sentiment(text) 
    return render_template('predict.html', data=sentiment, original_text=text)

if __name__ == '__main__':
    app.run(debug=True)