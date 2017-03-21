from __future__ import print_function
from mnist import evaluate
import sys

prediction, confidence = evaluate.from_local_image(sys.argv[1])
print("prediction: {}; confidence: {:.2f}".format(prediction, confidence))
