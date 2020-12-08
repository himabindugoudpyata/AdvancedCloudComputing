# Reference for kubeflow : https://github.com/kubeflow/examples/ 
# IMPORT LIBRARIES 
import logging
import argparse
import os
import uuid
from importlib import reload
from oauth2client.client import GoogleCredentials
credentials = GoogleCredentials.get_application_default()

import setup 
reload(setup)
setup.setup()

import k8s_util
import kubeflow
reload(kubeflow)
from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from kubeflow.tfjob.api import tf_job_client as tf_job_client_module
from IPython.core.display import display, HTML
import yaml

# Configure a Docker registry for Kubeflow Fairing¶
from kubernetes import client as k8s_client
from kubernetes.client import rest as k8s_rest
from kubeflow import fairing   
from kubeflow.fairing import utils as fairing_utils
from kubeflow.fairing.builders import append
from kubeflow.fairing.deployers import job
from kubeflow.fairing.preprocessors import base as base_preprocessor
from urllib.parse import urlencode

GCP_PROJECT = fairing.cloud.gcp.guess_project_name()
DOCKER_REGISTRY = 'gcr.io/{}/fairing-job'.format(GCP_PROJECT)
namespace = fairing_utils.get_current_k8s_namespace()
logging.info(f"Running in project {GCP_PROJECT}")
logging.info(f"Running in namespace {namespace}")
logging.info(f"Using Docker registry {DOCKER_REGISTRY}")
from kubeflow.fairing import constants
constants.constants.KANIKO_IMAGE = "gcr.io/kaniko-project/executor:v0.14.0"
from kubeflow.fairing.builders import cluster

parser = argparse.ArgumentParser()
parser.add_argument('--tf-model-script',
                      type=str,
                      default='kubeflow_mnist_model.py',
                      help='pick MNIST or the CIFAR10 script')

args = parser.parse_known_args()[0]

# output_map is a map of extra files to add to the notebook.
# It is a map from source location to the location inside the context.
output_map =  {
    "Dockerfile.model": "Dockerfile",
    "model.py": args.tf_model_script
}


preprocessor = base_preprocessor.BasePreProcessor(
    command=["python"], # The base class will set this.
    input_files=[],
    path_prefix="/app", # irrelevant since we aren't preprocessing any files
    output_map=output_map)

preprocessor.preprocess()

# Create a Cloud Storage bucket
from google.cloud import storage
bucket = f"{GCP_PROJECT}-model"

client = storage.Client()
b = storage.Bucket(client=client, name=bucket)

if not b.exists():
    logging.info(f"Creating bucket {bucket}")
    b.create()
else:
    logging.info(f"Bucket {bucket} already exists")    

cluster_builder = cluster.cluster.ClusterBuilder(registry=DOCKER_REGISTRY,
                                                 base_image="", 
                                                 preprocessor=preprocessor,
                                                 image_name=split(args.tf_model_script,'.')[0],
                                                 dockerfile_path="Dockerfile",
                                                 context_source=cluster.gcs_context.GCSContextSource())
cluster_builder.build()
logging.info(f"Built image {cluster_builder.image_tag}")

# Distributed Training 

train_name = f"model-train-{uuid.uuid4().hex[:4]}"
num_ps = 1
num_workers = 4
model_dir = f"gs://{bucket}/model"
export_path = f"gs://{bucket}/model/export" 
train_steps = 200
batch_size = 128
learning_rate = 0.01
image = cluster_builder.image_tag

train_spec = f"""apiVersion: kubeflow.org/v1
kind: TFJob
metadata:
  name: {train_name}  
spec:
  tfReplicaSpecs:
    Ps:
      replicas: {num_ps}
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          serviceAccount: default-editor
          containers:
          - name: tensorflow
            command:
            - python
            - /opt/model.py
            - --tf-model-dir={model_dir}
            - --tf-export-dir={export_path}
            - --tf-train-steps={train_steps}
            - --tf-batch-size={batch_size}
            - --tf-learning-rate={learning_rate}
            image: {image}
            workingDir: /opt
          restartPolicy: OnFailure
    Chief:
      replicas: 1
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          serviceAccount: default-editor
          containers:
          - name: tensorflow
            command:
            - python
            - /opt/model.py
            - --tf-model-dir={model_dir}
            - --tf-export-dir={export_path}
            - --tf-train-steps={train_steps}
            - --tf-batch-size={batch_size}
            - --tf-learning-rate={learning_rate}
            image: {image}
            workingDir: /opt
          restartPolicy: OnFailure
    Worker:
      replicas: 1
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: "false"
        spec:
          serviceAccount: default-editor
          containers:
          - name: tensorflow
            command:
            - python
            - /opt/model.py
            - --tf-model-dir={model_dir}
            - --tf-export-dir={export_path}
            - --tf-train-steps={train_steps}
            - --tf-batch-size={batch_size}
            - --tf-learning-rate={learning_rate}
            image: {image}
            workingDir: /opt
          restartPolicy: OnFailure
"""           

# Create the training job
tf_job_client = tf_job_client_module.TFJobClient()
tf_job_body = yaml.safe_load(train_spec)
tf_job = tf_job_client.create(tf_job_body, namespace=namespace)  
logging.info(f"Created job {namespace}.{train_name}")


for replica in ["chief", "worker", "ps"]:    
    logs_filter = f"""resource.type="k8s_container"    
    labels."k8s-pod/tf-job-name" = "{train_name}"
    labels."k8s-pod/tf-replica-type" = "{replica}"    
    resource.labels.container_name="tensorflow" """
    new_params = {'project': GCP_PROJECT,
                  'interval': 'P7D',
                  'advancedFilter': logs_filter}
    query = urlencode(new_params)
    url = "https://console.cloud.google.com/logs/viewer?" + query
    display(HTML(f"Link to: <a href='{url}'>{replica} logs</a>"))


# Check the status of the training and display logs
tf_job = tf_job_client.wait_for_condition(train_name, expected_condition=["Succeeded", "Failed"], namespace=namespace)

if tf_job_client.is_job_succeeded(train_name, namespace):
    logging.info(f"TFJob {namespace}.{train_name} succeeded")
else:
    raise ValueError(f"TFJob {namespace}.{train_name} failed")  

tf_job_client.get_logs(train_name, namespace)