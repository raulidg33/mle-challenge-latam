# MLE Challenge LATAM - Flight Delay Prediction API

This is an API built in python to predict whether a flight will be delayed or not using ML. This api is deployed in GCP App Engine.

## Usage

The API is hosted under the url given by GCP App Engine https://mle-challenge-latam.rj.r.appspot.com/ and has two endpoints: ```/health/``` and ```/predict/```.

### Get API health status
To get the api health status you can make a GET request to the ```/health/``` endpoint, which should return an OK value for status if the api is working correctly.

#### Example of request made with cURL:
```bash
curl -L https://mle-challenge-latam.rj.r.appspot.com/health/
```
Ths request will return the next json object:
```json
{
  "status": "OK"
}
```

### Predict flight delay

To use the trained model and predict a flights delay you can make a POST request to the ```/predict/``` endpoint, which will require a json object as an input like this one:
```json
{
  "flights": [
    {
      "OPERA": "Aerolineas Argentinas", 
      "TIPOVUELO": "N", 
      "MES": 3
    },
    ...
    {
      "OPERA": "Grupo LATAM", 
      "TIPOVUELO": "N", 
      "MES": 3
    }
  ]
}
```
#### Example of request made with cURL:

```bash
curl -X POST -d '{"flights": [{"OPERA": "Aerolineas Argentinas", "TIPOVUELO": "N", "MES": 3}]}' https://mle-challenge-latam.rj.r.appspot.com/predict
```
This request returns a json object like the next one:
```json
{
  "predict": [0]
}
```
Where the "predict" value is a list of the predicted labels, where 1 is delayed and 0 not delayed.