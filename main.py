from flask import Flask
from sklearn.neighbors import KNeighborsClassifier
import joblib
import numpy as np

knn = joblib.load("iris_modelo.pkl")

app = Flask(__name__)


@app.route("/")
def cad_flor():
	return("Botanico")

@app.route("/iris/predict/<sepal_length>/<sepal_width>/<petal_length>/<petal_width>/")

def pred_de_flores(sepal_length, sepal_width, petal_length, petal_width):
  
  a = [sepal_length, sepal_width, petal_length, petal_width]
  b = np.array(a, dtype=float)
  c = [float(i) for i in a]

  classe = knn.predict([c])

  return (classe[0])

app.run(host="0.0.0.0", port="8000")