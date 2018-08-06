# USAGE
# Start the server:
#   python run_keras_server.py
# Submit a request via cURL:
#   curl -X POST -F image=@dog.jpg 'http://localhost:5000/predict'
# Submita a request via Python:
#  python simple_request.py

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
from keras.models import load_model as k_load_model
from PIL import Image
import numpy as np
import flask
from flask import render_template
import io
import googleapiclient.discovery

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None

def load_model():
  # load the pre-trained Keras model (here we are using a model
  # pre-trained on ImageNet and provided by Keras, but you can
  # substitute in your own networks just as easily)
  global model
  model = k_load_model('./VGG16_data_augmented.h5')

def prepare_image(image, target):
  # if the image mode is not RGB, convert it
  if image.mode != "RGB":
    image = image.convert("RGB")

  # resize the input image and preprocess it
  image = image.resize(target)
  image = img_to_array(image)
  image = image.reshape((1,) + image.shape)
  image /= image.max() # rescale image
  print(image.shape)

  # return the processed image
  return image

@app.route("/predict", methods=["POST"])
def predict():
  # initialize the data dictionary that will be returned from the
  # view
  data = {"success": False}

  # ensure an image was properly uploaded to our endpoint
  if flask.request.method == "POST":
    if flask.request.files.get("image"):
      # read the image in PIL format
      image = flask.request.files["image"].read()
      image = Image.open(io.BytesIO(image))

      # preprocess the image and prepare it for classification
      image = prepare_image(image, target=(224, 224))

      # classify the input image and then initialize the list
      # of predictions to return to the client
      preds = model.predict(image)
      # TBD results = imagenet_utils.decode_predictions(preds)
      #data["predictions"] = []
      print(preds)
      data["predictions"] = preds.tolist()

      # loop over the results and add them to the list of
      # returned predictions
      #for (imagenetID, label, prob) in results[0]:
      #  r = {"label": label, "probability": float(prob)}
      #  data["predictions"].append(r)

      # indicate that the request was a success
      data["success"] = True

  # return the data dictionary as a JSON response
  return flask.jsonify(data)

@app.route("/predict", methods=["GET"])
def foo():
  return render_template('hello.html')

@app.route("/")
def index():
  return render_template('hello.html')

@app.route("/plant")
def plant():
  #def predict_json(project, instances, version=None):
  """Send json data to a deployed model for prediction.
  Args:
      project (str): project where the Cloud ML Engine Model is deployed.
      model (str): model name.
      instances ([[float]]): List of input instances, where each input
         instance is a list of floats.
      version: str, version of the model to target.
  Returns:
      Mapping[str: any]: dictionary of prediction results defined by the
          model.
  """
  project = 'keras-class-191806'
  model = 'plantDisease01'
  version=None
  # Create the ML Engine service object.
  # To authenticate set the environment variable
  # GOOGLE_APPLICATION_CREDENTIALS=<path_to_service_account_file>
  service = googleapiclient.discovery.build('ml', 'v1')
  name = 'projects/{}/models/{}'.format(project, model)

  if version is not None:
    name += '/versions/{}'.format(version)

  dummy = [0.2 for _ in range(7*7*512)]
  response = service.projects().predict(
    name=name,
    body={'input': dummy}
  ).execute()

  if 'error' in response:
    raise RuntimeError(response['error'])

  return response['predictions']

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
  print(("* Loading Keras model and Flask starting server..."
    "please wait until server has fully started"))
  load_model()
  app.run(debug=False, threaded=False)
