#website libraries
from flask import Flask, jsonify

#api libraries
import requests, json

#data processing libraries
from sklearn.preprocessing import MinMaxScaler

#machine learning model libraries
import keras
import numpy as np

app = Flask(__name__)

def getCloseData(symbol):
    headers = {"sec-fetch-site": "none",
        "sec-fetch-mode": "navigate",
        "sec-fetch-user":"?1",
        "sec-fetch-dest": "document",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36 Edg/110.0.1587.41",
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
    }
    url = "https://priceapi.moneycontrol.com/techCharts/indianMarket/stock/history?symbol="+symbol+"&resolution=60&from=1670089722&to=1676212159&countback=329&currencyCode=INR"

    response = requests.request("GET", url,headers = headers)

    eod = json.loads(response.text)
    return eod['c']


model = keras.models.load_model('LSTM_model')

def predictor(close_data,scaler):
  len_data = len(close_data)
  test_data = close_data[len_data - 60: , :]
  preds = []

  for i in range(7):
    x_test = [test_data]
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    pred = scaler.inverse_transform(predictions)
    preds = np.append(preds,pred[0][0])

    test_data = test_data[1:]
    test_data = np.append(test_data, predictions[0])
  return preds

@app.route('/')
def hello_world():
    return "hello there!!!"


@app.route('/predict/<stockName>',methods=['POST','GET'])
def predict(stockName):
    #//
    c = getCloseData(stockName)
    len_data = len(c)
    pred1 = c[-7:]
    new_data = []
    for i in c:
        new_data.append([i])
    
    scaler = MinMaxScaler(feature_range=(0,1))
    close_data = scaler.fit_transform(np.array(new_data))
    # Create the data sets x_test and y_tets
    preds = predictor(close_data,scaler)
    preds = pred1 + list(preds)
    data = {
        "name" : stockName,
        "preds" : preds
    }

    return jsonify(data)
    


if __name__ == '__main__':
    app.run(debug=True)