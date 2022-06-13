import pickle 
import numpy as np 
from mlflow.tracking import MlflowClient


client = MlflowClient()
experiment = client.get_experiment_by_name("MLModels")


print("Name: {}".format(experiment.name))
print("Experiment ID: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))


