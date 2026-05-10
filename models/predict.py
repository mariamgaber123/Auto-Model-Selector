# models/predict.py
import numpy as np

def predict_model(model, input_data):

    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

    if len(input_data.shape) == 1:
        input_data = input_data.reshape(1, -1)

    prediction = model.predict(input_data)

    if isinstance(prediction, (float, np.floating)):
        return round(float(prediction), 4)
    
    return prediction