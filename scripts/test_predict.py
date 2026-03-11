import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.predict import predict_svm

text = "Quán ăn rất ngon và phục vụ tốt"

print("Prediction:", predict_svm(text))