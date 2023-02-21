from flask import Flask, escape, request, render_template
import pickle


with open("vectorizer.pkl", 'rb') as file_object:
    vector = pickle.load(file_object)
with open("finalized_model.pkl", 'rb') as file_object:
    model = pickle.load(file_object)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print(news)

        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        return render_template("prediction.html", prediction_text="News headline is -> {}".format(predict))


    else:
        return render_template("prediction.html")


if __name__ == '__main__':
    app.debug = True
    app.run()