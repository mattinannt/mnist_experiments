from __future__ import print_function
from mnist import evaluate
import sys

prediction = evaluate.from_local_image(sys.argv[1])
print("prediction:", prediction)
