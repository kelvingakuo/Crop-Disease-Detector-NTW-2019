from azureml.core.conda_dependencies import CondaDependencies 

myenv = CondaDependencies()
myenv.add_pip_package("numpy")
myenv.add_pip_package("azureml-core")
myenv.add_pip_package("keras")

with open("predictor_env.yml","w") as f:
    f.write(myenv.serialize_to_string())