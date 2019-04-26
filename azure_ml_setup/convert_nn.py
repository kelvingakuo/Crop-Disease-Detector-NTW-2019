import keras2onnx
import onnx

from keras.models import load_model

nn = load_model('../trained_models/model.h5')

onnXAlexnet = keras2onnx.convert_keras(nn, 'onnxAlexnet',target_opset = 7)

onnx.save_model(onnXAlexnet, '../trained_models/model.onnx')