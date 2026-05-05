# models/predict.py
import numpy as np

def predict_model(model, input_data):

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)
    return prediction[0]