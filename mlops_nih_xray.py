# Databricks notebook source
# MAGIC %md
# MAGIC # From X-rays to a Production Classifier with MLflow
# MAGIC 
# MAGIC This simple example will demonstrate how to build a chest X-Ray classifer with PyTorch Lightning, and explain its output, but more importantly, will demonstrate how to manage the model's deployment to production as a REST service with MLflow and its Model Registry.
# MAGIC 
# MAGIC The National Institute of Health (NIH) [released a dataset](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community) of 45,000 chest X-rays of patients who may suffer from some problem in the chest cavity, along with several of 14 possible diagnoses. This was accompanied by a [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) analyzing the data set and presenting a classification model.
# MAGIC 
# MAGIC The task here is to train a classifier that learns to predict these diagnoses. Note that each image may have 0 or several 'labels'. This data set was the subject of a [Kaggle competition](https://www.kaggle.com/nih-chest-xrays/data) as well.

# COMMAND ----------

# MAGIC %pip install pytorch-lightning shap

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Engineering
# MAGIC 
# MAGIC The image data is provided as a series of [compressed archives](https://nihcc.app.box.com/v/ChestXray-NIHCC). However they are also available [from Kaggle](https://www.kaggle.com/nih-chest-xrays/data) with other useful information, like labels and bounding boxes. In this problem, only the images will be used, unpacked into an `.../images/` directory,, and the CSV file of label information `Data_Entry_2017.csv` at a `.../metadata/` path.
# MAGIC 
# MAGIC The images can be read directly and browsed with Spark:

# COMMAND ----------

raw_image_df = spark.read.format("image").load("/mnt/databricks-datasets-private/ML/nih_xray/images/")
display(raw_image_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Managing Unstructured Data with Delta
# MAGIC 
# MAGIC Although the images can be read directly as files, it will be useful to manage the data as a Delta table:
# MAGIC 
# MAGIC - Delta provides transactional updates, so that the data set can be updated, and still read safely while being updated
# MAGIC - Delta provides "time travel" to view previous states of the data set
# MAGIC - Reading batches of image data is more efficient from Delta than from many small files
# MAGIC - The image data needs some one-time preprocessing beforehand anyway
# MAGIC 
# MAGIC In this case, the images are all 1024 x 1024 grayscale images, though some arrive as 4-channel RGBA. They are normalized to 224 x 224 single-channel image data:

# COMMAND ----------

from pyspark.sql.types import BinaryType, StringType
from PIL import Image
import numpy as np

def to_grayscale(data, channels):
  np_array = np.array(data, dtype=np.uint8)
  if channels == 1: # assume mode = 0
    grayscale = np_array.reshape((1024,1024))
  else: # channels == 4 and mode == 24
    reshaped = np_array.reshape((1024,1024,4))
    # Data is BGRA; ignore alpha and use ITU BT.709 luma conversion:
    grayscale = (0.0722 * reshaped[:,:,0] + 0.7152 * reshaped[:,:,1] + 0.2126 * reshaped[:,:,2]).astype(np.uint8)
  # Use PIL to resize to match DL model that it will feed
  resized = Image.frombytes('L', (1024,1024), grayscale).resize((224,224), resample=Image.LANCZOS)
  # Flatten result into a bytearray
  return np.asarray(resized, dtype=np.uint8).flatten().tobytes()

to_grayscale_udf = udf(to_grayscale, BinaryType())

to_filename_udf = udf(lambda f: f.split("/")[-1], StringType())

image_df = raw_image_df.select(to_filename_udf("image.origin").alias("origin"), to_grayscale_udf("image.data", "image.nChannels").alias("image"))

# COMMAND ----------

# MAGIC %md
# MAGIC The file of metadata links the image file name to its labels. These are parsed and joined, written to a Delta table, and registered in the metastore:

# COMMAND ----------

raw_metadata_df = spark.read.\
  option("header", True).option("inferSchema", True).\
  csv("/mnt/databricks-datasets-private/ML/nih_xray/metadata/").\
  select("Image Index", "Finding Labels")

display(raw_metadata_df)

# COMMAND ----------

from pyspark.sql.functions import explode, split
from pyspark.sql.types import BooleanType, StructType, StructField

distinct_findings = sorted([r["col"] for r in raw_metadata_df.select(explode(split("Finding Labels", r"\|"))).distinct().collect() if r["col"] != "No Finding"])

encode_findings_schema = StructType([StructField(f.replace(" ", "_"), BooleanType(), False) for f in distinct_findings])

def encode_finding(raw_findings):
  findings = raw_findings.split("|")
  return [f in findings for f in distinct_findings]

encode_finding_udf = udf(encode_finding, encode_findings_schema)

metadata_df = raw_metadata_df.withColumn("encoded_findings", encode_finding_udf("Finding Labels")).select("Image Index", "encoded_findings.*")

table_path = "/tmp/sean.owen/image_table/"
metadata_df.join(image_df, metadata_df["Image Index"] == image_df["origin"]).drop("Image Index", "origin").write.mode("overwrite").format("delta").save(table_path)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS nih_xray;
# MAGIC USE nih_xray;
# MAGIC CREATE TABLE IF NOT EXISTS images USING DELTA LOCATION '/tmp/sean.owen/image_table/';

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modeling with PyTorch Lightning and MLflow
# MAGIC 
# MAGIC [PyTorch](https://pytorch.org/) is of course one of the most popular tools for building deep learning models, and is well suited to build a convolutional neural net that works well as a multi-label classifier for these images. Below, other related tools like `torchvision` and [PyTorch Lightning](https://www.pytorchlightning.ai/) are used to simplify expressing and building the classifier.
# MAGIC 
# MAGIC The data set isn't that large once preprocessed - about 2.2GB. For simplicity, the data will be loaded and manipulated with `pandas` from the Delta table, and model trained on one GPU. It's also quite possible to scale to multiple GPUs, or scale across machines with Spark and Horovod, but it won't be necessary to add that complexity in this example.

# COMMAND ----------

from sklearn.model_selection import train_test_split

df = spark.read.table("nih_xray.images")

train_pd, test_pd = train_test_split(df.toPandas(), test_size=0.1, random_state=42)

frac_positive = train_pd.drop("image", axis=1).sum().sum() / train_pd.drop("image", axis=1).size
disease_names = df.drop("image").columns
num_classes = len(disease_names)

# COMMAND ----------

# MAGIC %md
# MAGIC `torchvision` provides utilities that make it simple to perform some model-specific transformation as part of the model. Here, a pre-trained network will be used which requires normalized 3-channel RGB data as PyTorch Tensors:

# COMMAND ----------

from torchvision import transforms

transforms = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Lambda(lambda image: image.convert('RGB')),
  transforms.ToTensor(),
  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# COMMAND ----------

# MAGIC %md
# MAGIC Define the `Dataset` and train/test `DataLoader`s for this data set in PyTorch:

# COMMAND ----------

from torch.utils.data import Dataset, DataLoader
import numpy as np

class XRayDataset(Dataset):
  def __init__(self, data_pd, transforms):
    self.data_pd = data_pd
    self.transforms = transforms
    
  def __len__(self):
    return len(self.data_pd)
  
  def __getitem__(self, idx):
    image = np.frombuffer(self.data_pd["image"].iloc[idx], dtype=np.uint8).reshape((224,224))
    labels = self.data_pd.drop("image", axis=1).iloc[idx].values.astype(np.float32)
    return self.transforms(image), labels

train_loader = DataLoader(XRayDataset(train_pd, transforms), batch_size=64, num_workers=8, shuffle=True)
test_loader = DataLoader(XRayDataset(test_pd, transforms), batch_size=64, num_workers=8)

# COMMAND ----------

# MAGIC %md
# MAGIC Note that MLflow natively supports logging PyTorch models of course, but, can also automatically log the output of models defined with PyTorch Lightning:

# COMMAND ----------

import mlflow.pytorch

mlflow.pytorch.autolog()

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, the model is defined, and fit. For simple purposes here, the model itself is quite simple: it employs the pretrained [densenet121](https://pytorch.org/hub/pytorch_vision_densenet/) layers to do most of the work (layers which are not further trained here), and simply adds some dropout and a dense layer on top to perform the classification. No attempt is made here to tune the network's architecture or parameters further.
# MAGIC 
# MAGIC For those new to PyTorch Lightning, it is still "PyTorch", but removes the need to write much of PyTorch's boilerplate code. Instead, a `LightningModule` class is implemented with key portions like model definition and fitting processes defined.
# MAGIC 
# MAGIC _Note: This section should be run on a GPU. An NVIDIA T4 GPU is recommended, though any modern GPU should work. This code can also be easily changed to train on CPUs or TPUs._

# COMMAND ----------

import torch
from torch.optim import Adam
from torch.nn import Dropout, Linear
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.metrics import log_loss
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class XRayNNLightning(pl.LightningModule):
  def __init__(self, learning_rate, pos_weights):
    super(XRayNNLightning, self).__init__()
    self.densenet = torch.hub.load('pytorch/vision:v0.6.0', 'densenet121', pretrained=True)
    for param in self.densenet.parameters():
      param.requires_grad = False
    self.dropout = Dropout(0.5)
    self.linear = Linear(1000, num_classes)
    # No sigmoid here; output logits
    self.learning_rate = learning_rate
    self.pos_weights = pos_weights

  def get_densenet():
    return self.densenet
    
  def forward(self, x):
    x = self.densenet(x)
    x = self.dropout(x)
    x = self.linear(x)
    return x

  def configure_optimizers(self):
    return Adam(self.parameters(), lr=self.learning_rate)

  def training_step(self, train_batch, batch_idx):
    x, y = train_batch
    output = self.forward(x)
    # Outputting logits above lets us use binary_cross_entropy_with_logits for efficiency, but also, allows the use of
    # pos_weight to express that positive labels should be given much more weight. 
    # Note this was also proposed in the paper linked above.
    loss = binary_cross_entropy_with_logits(output, y, pos_weight=torch.tensor(self.pos_weights).to(self.device))
    self.log('train_loss', loss)
    return loss

  def validation_step(self, val_batch, batch_idx):
    x, y = val_batch
    output = self.forward(x)
    val_loss = binary_cross_entropy_with_logits(output, y, pos_weight=torch.tensor(self.pos_weights).to(self.device))
    self.log('val_loss', val_loss)

model = XRayNNLightning(learning_rate=0.001, pos_weights=[[1.0 / frac_positive] * num_classes])

# Let PyTorch handle learning rate, batch size tuning, as well as early stopping.
# Change here to configure for CPUs or TPUs.
trainer = pl.Trainer(gpus=1, max_epochs=100, 
                     auto_scale_batch_size='binsearch',
                     auto_lr_find=True,
                     callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=True)])
trainer.fit(model, train_loader, test_loader)

# COMMAND ----------

# MAGIC %md
# MAGIC Although not shown here for brevity, this model's results are comparable to those cited in the [paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf) - about 0.6-0.7 AUC for each of the 14 classes. The auto-logged results are available in MLflow:
# MAGIC 
# MAGIC <img src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/nih_xray/pytorch_params.png" width="600"/> <img src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/nih_xray/pytorch_artifacts.png" width="600"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serving the Model with MLflow
# MAGIC 
# MAGIC This auto-logged model is useful raw material. The goal is to deploy it as a REST API, and [MLflow can create a REST API and Docker container](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models) around a `pyfunc` model, and even deploy to Azure ML or AWS SageMaker for you. It can also be deployed within Databricks for testing.
# MAGIC 
# MAGIC However, there are a few catches which mean we can't directly deploy the model above:
# MAGIC 
# MAGIC - It accepts images as input, but these can't be directly specified in the JSON request to the REST API
# MAGIC - Its output are logits, when probabilities (and label names) would be more useful
# MAGIC 
# MAGIC It is however easy to define a custom `PythonModel` that will wrap the PyTorch model and perform additional pre- and post-processing. This model accepts a base64-encoded image file, and returns the probability each label:

# COMMAND ----------

import torch
import pandas as pd
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from mlflow.pyfunc import PythonModel

class XRayNNServingModel(PythonModel):
  def __init__(self, model, transforms, disease_names):
    self.model = model
    self.transforms = transforms
    self.disease_names = disease_names
    
  def get_model():
    return self.model
  
  def get_transforms():
    return self.transforms
  
  def get_disease_names():
    return disease_names

  def predict(self, context, model_input):
    def infer(b64_string):
      encoded_image = base64.decodebytes(bytearray(b64_string, encoding="utf8"))
      image = Image.open(BytesIO(encoded_image)).convert(mode='L').resize((224,224), resample=Image.LANCZOS)
      image_bytes = np.asarray(image, dtype=np.uint8)
      transformed = self.transforms(image_bytes).unsqueeze(dim=0)
      output = self.model(transformed).squeeze()
      return torch.sigmoid(output).tolist()
    return pd.DataFrame(model_input.iloc[:,0].apply(infer).to_list(), columns=disease_names)

# COMMAND ----------

# MAGIC %md
# MAGIC Now the new wrapped model is logged with MLflow:

# COMMAND ----------

import mlflow.pyfunc
import mlflow.pytorch
import mlflow.models
import pytorch_lightning as pl
import PIL
import torchvision

# Load PyTorch Lightning model
model = mlflow.pytorch.load_model("runs:/fcc38232b1a84b878be1d5877ba206b5/model", map_location='cpu')

with mlflow.start_run():
  model_env = mlflow.pyfunc.get_default_conda_env()
  # Record specific additional dependencies required by the serving model
  model_env['dependencies'][-1]['pip'] += [
    f'torch=={torch.__version__}',
    f'torchvision=={torchvision.__version__}',
    f'pytorch-lightning=={pl.__version__}',
    f'pillow=={PIL.__version__}',
  ]
  # Log the model signature - just creates some dummy data of the right type to infer from
  signature = mlflow.models.infer_signature(
    pd.DataFrame(["dummy"], columns=["image"]),
    pd.DataFrame([[0.0] * num_classes], columns=disease_names))
  python_model = XRayNNServingModel(model, transforms, disease_names)
  mlflow.pyfunc.log_model("model", python_model=python_model, signature=signature, conda_env=model_env)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Registering the Model
# MAGIC 
# MAGIC The MLflow Model Registry provides workflow management for the model promotion process, from Staging to Production. The new run created above can be registered directly from the MLflow UI:
# MAGIC 
# MAGIC <img src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/nih_xray/register_model.png" width="800"/>
# MAGIC 
# MAGIC It can then be transitioned into the Production state directly, for simple purposes here. After that, enabling serving within Databricks is as simple as turning it on in the models' Serving tab:
# MAGIC 
# MAGIC <img src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/nih_xray/serving.png" width="800"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Accessing the Model with a REST Request
# MAGIC 
# MAGIC Now, we can send images to the REST endpoint and observe its classifications. This could power a simple web application, but here, to demonstrate, it is called directly from a notebook.

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text("Path", "/dbfs/mnt/databricks-datasets-private/ML/nih_xray/images/00000001_000.png", "Path")

# COMMAND ----------

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = dbutils.widgets.get("Path")
plt.imshow(mpimg.imread(image_path), cmap='gray')

# COMMAND ----------

import base64
import requests
import pandas as pd

with open(image_path, "rb") as file:
  content = file.read()

dataset = pd.DataFrame([base64.encodebytes(content)], columns=["image"])
# Note that you will still need a Databricks access token to send with the request. This can/should be stored as a secret in the workspace:
token = dbutils.secrets.get("demo-token-sean.owen", "token")

response = requests.request(method='POST',
                            headers={'Authorization': f'Bearer {token}'}, 
                            url='https://demo.cloud.databricks.com/model/nih_xray/2/invocations',
                            json=dataset.to_dict(orient='split'))
pd.DataFrame(response.json())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The model suggests that a doctor might examine this X-ray for Atelectasis and Consolidation, but a Hernia is unlikely, for example.
# MAGIC But, why did the model think so? Fortunately there are tools that can explain the model's output in this case, and this will be demonstrated a little later.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Adding Webhooks for Model State Management
# MAGIC 
# MAGIC MLflow can now trigger webhooks when Model Registry events happen. Webhooks are standard 'callbacks' which let applications signal one another. For example, a webhook can cause a CI/CD test job to start and run tests on a model. In this simple example, we'll just set up a webhook that posts a message to a Slack channel.
# MAGIC 
# MAGIC _Note_: the example below requires a [registered Slack webhook](https://api.slack.com/messaging/webhooks). Because the webhook URL is sensitive, it is stored as a secret in the workspace and not included inline.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.utils.rest_utils import http_request
import json

def mlflow_call_endpoint(endpoint, method, body = '{}'):
  client = MlflowClient()
  host_creds = client._tracking_client.store.get_host_creds()
  if method == 'GET':
    response = http_request(host_creds=host_creds, endpoint=f"/api/2.0/mlflow/{endpoint}", method=method, params=json.loads(body))
  else:
    response = http_request(host_creds=host_creds, endpoint=f"/api/2.0/mlflow/{endpoint}", method=method, json=json.loads(body))
  return response.json()

json_obj = {
  "model_name": "nih_xray",
  "events": ["MODEL_VERSION_CREATED", "TRANSITION_REQUEST_CREATED", "MODEL_VERSION_TRANSITIONED_STAGE", "COMMENT_CREATED", "MODEL_VERSION_TAG_SET"],
  "http_url_spec": { "url": dbutils.secrets.get("demo-token-sean.owen", "slack_webhook") }
}
mlflow_call_endpoint("registry-webhooks/create", "POST", body=json.dumps(json_obj))

# COMMAND ----------

# MAGIC %md
# MAGIC As model versions are added, transitioned among stages, commented on, etc. a webhook will fire.
# MAGIC 
# MAGIC <img src="https://databricks-knowledge-repo-images.s3.us-east-2.amazonaws.com/ML/nih_xray/slack.png" width="600"/>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explaining Predictions
# MAGIC 
# MAGIC [SHAP](https://shap.readthedocs.io/en/latest/) is a popular tool for explaining model predictions. It can explain virtually any classifier or regressor at the prediction level, and estimate how much each input feature contributed positively or negatively to the result, and by how much.
# MAGIC 
# MAGIC In MLflow 1.12 and later, SHAP model explanations can be [logged automatically](https://www.mlflow.org/docs/latest/python_api/mlflow.shap.html):
# MAGIC 
# MAGIC <img src="https://www.mlflow.org/docs/latest/_images/shap-ui-screenshot.png" width="600"/>
# MAGIC 
# MAGIC However, this model's inputs are not simple scalar features, but an image. SHAP does have tools like `GradExplainer` and `DeepExplainer` that are specifically designed to explain neural nets' classification of images. To use this, we do have to use SHAP manually instead of via MLflow's automated tools. However the result can be, for example, logged with a model in MLflow.
# MAGIC 
# MAGIC Here we explain the model's top classification, and generate a plot showing which parts of the image most strongly move the prediction positively (red) or negatively (blue). The explanation is traced back to an early intermediate layer of densenet121.

# COMMAND ----------

import numpy as np
import torch
import mlflow.pyfunc
import shap

# Load the latest production model and its components
pyfunc_model = mlflow.pyfunc.load_model("models:/nih_xray/production")
transforms = pyfunc_model._model_impl.python_model.transforms
model = pyfunc_model._model_impl.python_model.model
disease_names = pyfunc_model._model_impl.python_model.disease_names

# Let's pick an example that definitely exhibits some affliction, like Effusion
df = spark.read.table("nih_xray.images")
first_row = df.filter("Effusion").select("image").limit(1).toPandas()
image = np.frombuffer(first_row["image"].item(), dtype=np.uint8).reshape((224,224))

# Only need a small sample for explanations
sample = df.sample(0.05).select("image").toPandas()
sample_tensor = torch.cat([transforms(np.frombuffer(sample["image"].iloc[idx], dtype=np.uint8).reshape((224,224))).unsqueeze(dim=0) for idx in range(len(sample))])

e = shap.GradientExplainer((model, model.densenet.features[6]), sample_tensor, local_smoothing=0.1)
shap_values, indexes = e.shap_values(transforms(image).unsqueeze(dim=0), ranked_outputs=3, nsamples=500)

shap.image_plot(shap_values[0][0].mean(axis=0, keepdims=True),
                transforms(image).numpy().mean(axis=0, keepdims=True))

# COMMAND ----------

pd.DataFrame(torch.sigmoid(model(transforms(image).unsqueeze(dim=0))).detach().numpy(), columns=disease_names).iloc[:,indexes.numpy()[0]]

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC This suggests that the small region at the top of left lung is more significant in causing the model to produce its positive classifications for Infiltration, Effusion and Cardiomegaly than most of the image, and the bottom of the left lung however contradicts those to some degree and is associated with lower probability of that classification.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Managing Notebooks with Projects
# MAGIC 
# MAGIC This notebook exists within a Project. This means it and any related notebooks are backed by a Git repository. The notebook can be committed, along with other notebooks, and observed in the source Git repository.

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### TODO