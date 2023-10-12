# tensorflow-cpu-multiworkers
To demonstrate how to create a Custom Job on Vertex AI using CPU multi-worker strategy

## Source Code
The origin code for this demo is from <b>Prototype to Production:Distributed training on Vertex AI</b> by Nikita Namjoshi on codelabs.
My intention is to modify the code in order to:
- handle training when no GPU is available - find an equivalent alternative to achieve the task
- demonstrate multi-worker strategy for tensorflow can be applied to CPU-only workers
- create and monitor a custom job on Vertex AI

These all are tasks for a Machine Learning Engineer who has different skill set and focus regarding AI/ML.

## Step 0
As mentioned in the original repo, you need to set up your Google Cloud environment properly. 
Your GCP/Vertex AI Platform environment can be different from mine but the pre-requisites are the same : a GCP Project, IAM permissions for yourself or any Service Account, Artifactory Registry, Cloud Storage Bucket with the flower image dataset and Jupyter Notebook (no GPU is required / indeed GPU cannot be attached to a notebook instance while you are on free trial)

## Step 1
Create a Dockerfile locally and build the docker image and push it to Artifact Registry.

The original Dockerfile uses a base image with GPU capability. To use CPU only on multi-worker strategy, you need select the other options. A full list of base images can be found as follows:

[Google DeepLearning Container](https://cloud.google.com/deep-learning-containers/docs/choosing-container)

The image in my Dockerfile is:
gcr.io/deeplearning-platform-release/tf2-cpu.2-12.py310

Docker build command is simple and straightforward for local build:

<b>docker build ./ -t us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/flower_image_distributed:multi_machine</b>

Remember to use variables to match your own PROJECT_ID and Artifact Registry REPO_NAME.
Please also make sure the tag: multi_machine or any unique tag you prefer is used

## Step 2
As indicated in the Dockerfile, you need to create a trainer directory with task.py undeer it since the entrypoint for the container will be the task.py as a module (task) under trainer (trainer/task).

The task.py is based on Point 5 of the origin repo, several functions are added for multiple workers and the most important one is to ensure the strategy used will be : MultiWorkerMirroredStrategy()

## Step 3
When your image and task.py python file containing the CNN model and flower images (Training Data) stored under your GCS Bucket (BUCKET_ROOT in task.py) are all ready, you can start your custom training on the Jupyter notebook.

On your notebook instance/ JupyterLab console, open the Tensorflow 2 (local) Launcher to create a notebook flower-multi.ipynb:

- there are only 3 cells
- the 1st is to import aiplatform
- the 2nd is to create a worker_pool_specs - the example does 2 workers using the cpu-only image with tag multi-machine
- the 3rd is to create and run a custom job on Vertex AI using MultiWorkerMirroredStrategy
  - MultiWorkerMirroredStrategy is from the TF model in task.py main function
  - 2 workers as specified in worker_pool_specs
  - one of the worker will be assigned to be Chief - to be responsible for syncing parameters and writing the output SavedModel to Cloud Storage

## Step 4
After successful run of all cells, a Custom Job on Vertex AI is created.
It will be queued and prepared for provisioning, some waiting time is excepted.
You should get the status from tne console as well as from Logs Explorer.
The latter gives you more information regarding the progress of training, information and error of the job, lest it is not real-time basis. Still it is the best source for troubleshooting any problems.
If everything is fine, the training job will be completed and the Saved_Model (in protocolbuf) and other artifacts will be saved to Cloud Storage (or you can save it on Artifact Registry - which is the better option)



![image](https://github.com/tonyktchan/tensorflow-cpu-multiworkers/assets/96426553/7bf5cd1f-7325-4fa7-8087-2f57ac88a956)


and the SavedModel can be found under the 'multi-machine-output' folder (indeed just a dummy folder for object storage):



![image](https://github.com/tonyktchan/tensorflow-cpu-multiworkers/assets/96426553/a4073ae2-98e4-47c0-a4eb-ab80465771d9)


By then, your ML model is ready to be deployed!


    

