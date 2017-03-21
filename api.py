from flask import Flask, jsonify, request
import numpy as np
from mnist import evaluate
import urllib.request

# new flask app
app = Flask(__name__)


@app.route("/api/mnist", methods=["POST"])
def mnist():
    input = request.json
    image_url = input['imageUrl']
    destination = '/tmp/image.png'
    urllib.request.urlretrieve(image_url, destination)
    prediction, confidence = evaluate.from_local_image(destination)
    return jsonify(prediction=str(prediction), confidence=str(confidence))


if __name__ == "__main__":
    app.run()
