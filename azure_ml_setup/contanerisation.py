from azureml.core.image import ContainerImage
from azureml.core.model import Model
from azureml.core.webservice import Webservice
from azureml.core import Workspace


image_config = ContainerImage.image_configuration(execution_script = "score.py", runtime = "python",conda_file = "predictor_env.yml",description = "Environment definitions")

service = Webservice.deploy_from_model(workspace=ws, name='Agrix-Predictions', models=[model], image_config=image_config)

service.wait_for_deployment(show_output=True)

print(service.scoring_uri)