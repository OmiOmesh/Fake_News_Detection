from flask import Flask, request, render_template, send_from_directory
import os
import pickle

vector = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('finalModel.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['Get', 'POST'])
def prediction():
    if request.method == 'POST':
        news = str(request.form['news'])
        print(news)
        predict = model.predict(vector.transform([news]))[0]
        print(predict)

        return render_template('prediction.html', prediction_text="News Headline is -> {}". format(predict))
    else:
        return render_template('prediction.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    root_dir = os.path.dirname(os.getcwd())
    return send_from_directory(os.path.join(root_dir, 'static'), filename)

if __name__ == '__main__':
    app.run(debug=True)