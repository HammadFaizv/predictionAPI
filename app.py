#website libraries
from flask import Flask,request, url_for, redirect, render_template, jsonify

#api libraries
import requests, json, urllib.request

#news processing libraries
from training_data import prediction,data_cleaner

#machine learning model libraries
import joblib
import numpy as np

#time input
import datetime

app = Flask(__name__)

model=joblib.load('StockLogReg.pkl')
date = datetime.datetime.now()
today_date = str(date.year)+'-'+str(date.month)+'-'+str(date.day)

def model_prediction(data):
    data = prediction(data)
    return model.predict_proba(data[:,:584289])

@app.route('/')
def hello_world():
    return "hello there!!!"


@app.route('/predict/<stockName>',methods=['POST','GET'])
def predict(stockName):
    stockName = str(stockName)
    key = '7af6ce7078524ceeb098006f32cbdd95'
    url = 'https://newsapi.org/v2/everything?q='+ stockName +'&from='+ today_date +'&sortBy=publishedAt&apiKey='+key
    url_req = urllib.request.urlopen(url)
    news = json.load(url_req)
    news_headlines = news.get("articles")
    headlines = []
    for i in range(0,len(news_headlines)):
        headlines.append(news_headlines[i].get("description"))

    train_data = data_cleaner(headlines)
    pred = model_prediction(train_data)
    #return render_template("forest.html",pred=pred[0][0])
    preds = {
        "id" : stockName,
        "prediction" : pred[0][1]
    }

    return jsonify(preds)


if __name__ == '__main__':
    app.run(debug=True)


#api : 7af6ce7078524ceeb098006f32cbdd95