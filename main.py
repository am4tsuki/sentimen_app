import pickle
import numpy as np
from flask import Flask, render_template, request

# initiate flask app
app = Flask(__name__)
# load tokopedia model
nb = pickle.load(open('./model/naivebayes.pkl', 'rb'))
vectorize = pickle.load(open('./model/vectorizer.pkl', 'rb'))
# index route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        title = 'Analisis Sentimen Google Play Store'
        return render_template('index.html', title=title)
    if request.method == 'POST':
        title = 'Analisis Sentimen Google Play Store'
        review = request.form.get('review')
        vector = vectorize.transform([review])
        result = nb.predict(vector)
        proba = nb.predict_proba(vector)
        proba = np.max(proba)
        percent = round(proba*100)
        if result == 1:
            review = 'Positif'
        else:
            review = 'Negatif'
        return render_template('result.html',title=title, proba=proba, percent=percent, review=review.capitalize())
@app.route('/team')
def team():
    title = 'Analisis Sentimen Google Play Store'
    return render_template('team.html', title=title)
if __name__ == '__main__':
    app.run(debug=True)