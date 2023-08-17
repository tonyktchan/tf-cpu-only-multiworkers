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

docker build ./ -t us-central1-docker.pkg.dev/$PROJECT_ID/$REPO_NAME/flower_image_distributed:multi_machine

Remember to use variables to match your own PROJECT_ID and Artifact Registry REPO_NAME.
Please also make sure the tag: multi_machine or any unique tag you prefer is used

## Step 2
As indicated in the Dockerfile, you need to create a trainer directory with task.py undeer it since the entrypoint for the container will be the task.py as a module (task) under trainer (trainer/task).

The task.py is the same as from Point 5 of the origin repo, it is the same for either a single worker or multiple workers.

## Step 3
When your image and task.py python file containing the CNN model and flower images (Training Data) are all ready, you can start your custom training on the Jupyter notebook.

On your notebook instance/ JupyterLab console, open the Tensorflow 2 (local) Launcher to create a notebook flower-multi.ipynb:

- there are only 3 cells
- the 1st is to import aiplatform
- the 2nd is to create a worker_pool_specs - the example does 2 workers using the cpu-only image with tag multi-machine
- the 3rd is to create and run a custom job on Vertex AI using MultiWorkerMirroredStrategy
  - MultiWorkerMirroredStrategy is from the TF model in task.py main function
  - 2 workers as specified in worker_pool_specs
 
    

