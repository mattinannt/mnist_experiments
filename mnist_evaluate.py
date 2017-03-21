from __future__ import print_function
from mnist import evaluate
import sys

model = evaluate.init()

prediction1, confidence1 = evaluate.from_local_image(sys.argv[1], model)
print("prediction: {}; confidence: {:.2f}".format(prediction1, confidence1))
prediction2, confidence2 = evaluate.from_local_image(sys.argv[1], model)
print("prediction: {}; confidence: {:.2f}".format(prediction2, confidence2))
