from __future__ import print_function
from mnist import evaluate
import sys

prediction = evaluate.run(sys.argv[1])
print("prediction:", prediction)
