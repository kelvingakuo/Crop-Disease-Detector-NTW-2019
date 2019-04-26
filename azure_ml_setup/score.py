from azureml.core.model import Model
import numpy as np

import onnxruntime



def init():
    global model_path
    global classes

    model_path = Model.get_model_path(model_name = 'Predictor')
    

    session = onnxruntime.InferenceSession("keras_example.onnx")

    first_input_name = session.get_inputs()[0].name

    first_output_name = session.get_outputs()[0].name

    

def run(raw_data):
    try:
        data = np.array(json.loads(raw_data)['data'])
        
        results = session.run([first_output_name], {first_input_name: data})

        result = results[0].tolist()

        return {"result": result}
    except Exception as e:
        result = str(e)
        return {"error": result}