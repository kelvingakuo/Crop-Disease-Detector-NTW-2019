from azureml.core import Workspace
from azureml.core.model import Model


# Create Workspace
ws = Workspace.create(name = 'tomato_pepper_potato_predictor', subscription_id = '94309b5e-965a-4b8c-850a-95a8824ee3a7', create_resource_group = True, location = 'westeurope')
# Register model with workspace
model = Model.register(model_path = '../trained_models/model.onnx', model_name = 'Tomato, Pepper, and potato Predictor', description = 'Some model trained to detect tomato, potato, and pepper diseases', workspace = ws)