from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("home.html")

@app.route("/home", methods=['POST'])
def home():
    text = request.form['text']
    #sentiment = model_predict(text) 

    #return render_template('predict.html', data=sentiment)
    return render_template("predict_html")

if __name__ == '__main__':
    app.run(debug=True)