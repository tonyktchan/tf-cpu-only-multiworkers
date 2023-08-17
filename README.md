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
Your GCP/Vertex AI Platform environment can be different from mine but the pre-requisites are the same : a GCP Project, IAM permissions for yourself or any Service Account, Artifactory Registry, Cloud Storage Bucket and Jupyter Notebook (no GPU is required / indeed GPU cannot be attached to a notebook instance while you are on free trial)

## step 1
Create a Dockerfile locally and build the docker image and push it to Artifact Registry.

The original Dockerfile uses a base image with GPU capability. To use CPU only on multi-worker strategy, you need select the other options. A full list of base images can be found as follows:

[Google DeepLearning Container](https://cloud.google.com/deep-learning-containers/docs/choosing-container)

