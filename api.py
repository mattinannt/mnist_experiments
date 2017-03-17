from flask import Flask, jsonify, request
import numpy as np
from mnist import evaluate

# new flask app
app = Flask(__name__)

@app.route("/api/mnist", methods=["POST"])
def mnist():
    input = ((255 - np.array(request.json, dtype=np.uint8)) / 255.0).reshape(1, 784)
    output = evaluate.run(input)
    return jsonify(prediction=output)

if __name__ == "__main__":
    app.run()
