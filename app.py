from flask import Flask, jsonify, request
import numpy as np
from sklearn.externals import joblib
import pandas as pd
from sklearn.linear_model import SGDClassifier
import numpy as np
from sklearn import linear_model
from sklearn.externals import joblib
from bs4 import BeautifulSoup
import re
from sklearn.feature_extraction.text import CountVectorizer
import praw
import os
import json

#file upload library
from werkzeug import secure_filename

# https://www.tutorialspoint.com/flask
import flask
app = Flask(__name__)
UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = {'txt'}
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd() , "uploads")
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

reddit = praw.Reddit(client_id='#', client_secret='#',
                     user_agent='#', username='Aman_Sawarn', password='#')

###################################################
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


stopwords= set(['br', 'the', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",\
            "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', \
            'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their',\
            'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', \
            'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', \
            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', \
            'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',\
            'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',\
            'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',\
            'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', \
            's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', \
            've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',\
            "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',\
            "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", \
            'won', "won't", 'wouldn', "wouldn't"])


def detect_flair(url):
    submission = reddit.submission(url=url)
    
    data = {}
    data['title'] = submission.title
    data['url'] = submission.url
    submission.comments.replace_more(limit=None)
    comment = ''
    for top_level_comment in submission.comments:
        comment = comment + ' ' + top_level_comment.body
    
    data["comment"] = comment
    data['title'] = decontracted(data['title'])
    data['comment'] = decontracted(data['comment'])
    data['combine'] = data['title'] + data['comment'] + data['url']
    
    return data['combine']
###################################################

def predictSingle(string):
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    review_text = decontracted(string)
#     print(review_text)
    test_vect = count_vect.transform(([review_text]))
    pred = clf.predict(test_vect)

    return pred

@app.route('/')
def hello_world():
    return 'Please make a POST request at /automated_testing with a text file containing URLs!'


@app.route('/index')
def index():
    return flask.render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
#def predict(string):
    string = request.form['message']
    clf = joblib.load('model.pkl')
    count_vect = joblib.load('count_vect.pkl')
    string=detect_flair(string)
    review_text = decontracted(string)
    test_vect = count_vect.transform(([review_text]))
    pred = clf.predict(test_vect)
    print(pred)
    return flask.render_template('result.html', prediction_text='{}'.format(pred))
   # return jsonify({'prediction': pred})

@app.route('/automated_testing' , methods=['POST'])
def test():
    print(flask.request.files)
    file = flask.request.files['upload_file']
    fileName = secure_filename(file.filename)
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileName))
    f = open(os.path.join(app.config['UPLOAD_FOLDER'], fileName))
    textTostring = f.read().split("\n")
    fileResponse = {}
    for oneString in textTostring:
        if not len(oneString)<8:
            print(oneString)
            print(predictSingle(detect_flair(oneString)))
            fileResponse[oneString] = predictSingle(detect_flair(oneString))[0]
    return  json.dumps(fileResponse, cls=NumpyArrayEncoder)


# https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)





@app.route('/automated_testing' , methods = ['GET'])
def testGet():
    return flask.render_template('test.html')




if __name__ == '__main__':
    app.run(debug=True)
