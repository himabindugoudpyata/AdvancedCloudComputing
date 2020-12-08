### AdvancedCloudComputing 

### Project Name : Performance Evaluation of Machine Learning Algorithms Using Fiber and Kubeflow

#### Contributors
* Himabindu Goud Pyata (hp93920@uga.edu)
* Anagha Narasimha Joshi (Anagha.Joshi@uga.edu)

#### Cloud Technologies used
* Google Cloud Platform
* Kubernetes & Kubeflow
* Docker
* Uber-Fiber

#### Motivation 
M​achine learning algorithms require a large amount of data to do incremental improvements in the performance. However, it is nearly impossible for a local computer to process the vast amount of data. As a result, practitioners use distributed computing for obtaining high-computational power to deliver quick and accurate results.However, there are quite a few challenges in using it. Some of them are :

* There is still a huge gap between making code work locally on laptops or desktops and running code on a production cluster.
* Dynamic scaling is unreliable
* High learning cost

These issues may cause delay in training and evaluating models at scale.To address these challenges,there are many frameworks and tools that are available today. In this project , our aim is to evaluate and compare two of them namely Fiber (Uber open-sourced) and Kubeflow.
Fiber is a new Python distributed library (open-sourced). It is designed to enable users to implement large scale computation easily on a computer cluster.The aim is to efficiently support applications on a large amount of diverse computing hardware, dynamically scale algorithms to improve resource usage efficiency, and reduce the engineering burden required to make complex algorithms work on computer clusters.Where as ,​Kubeflow is a free, open-source machine learning platform that makes it possible for machine learning pipelines to orchestrate complicated workflows runningonKubernetes.It easily enables the poweroftrainingmachinelearningmodels on multiple computers, accelerating the time to train a model.

### Methodology:
 
 A two phase approach has been taken to evaluate the performance of neural network algorithms on Fiber and KubeFlow which is described as follows:

1) Two  different  Datasets/  Problem  types  were chosen for this study:
    a)  MNIST  Handwritten  digits  Recognition Problem(which  has  over  60000  instances):  Here  the  goal is  to    look  at  an  input  image  and  recognize  which digit is drawn in it.
    b)  CIFAR-10  Object  Classification  problem  (which has over 50000 instances): Here the goal is to look at  images  of  objects  and  decide  which  of  the  10 classes  (ship,plane  etc)  this  image  belongs  to  and categorize it accordingly.

2)  Both  problem  statements  were  solved  with  the  help  of 2 layer neural networks and deployed on Google   Cloud Platform.

The  comparison  of  the  frameworks  (Fiber,  KubeFlow)  in this study is based on the following parameters:

. CPU Utilization: It denotes the percentage of CPU that was used while the neural network were being executed on the dataset.

. Training Time: As neural networks are considered, first the  model  is  trained  using  training  dataset.  Time  taken to train the model is called training time. This varies onthe implementations of the algorithms.

.Throughput: Decided by the Peak Read and Write times.From  above  metrics  by  comparing  ,we  can  decide  which framework is effective and efficient on larger datasets.

#### Experiment Apparatus

Number of CPUs : 8
RAM : 32GB
number of epochs = 10
Mini batch size = 128 

##### Datasets 

The MNIST database of handwritten digits, available from this page, has a training set of 60,000 examples, and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image.

The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.


##### Model Architecture (2 layered CNN )
-------------------------------------------------------------------
Layer (type)         |        Output Shape       |       Param #   |
|--------------------|---------------------------|-----------------|
conv2d_2 (Conv2D)      |      (None, 28, 28, 32)   |     832       |
max_pooling2d_2 (MaxPooling2 |(None, 14, 14, 32)   |     0    |     
conv2d_3 (Conv2D)         |   (None, 14, 14, 64)    |    51264   |  
max_pooling2d_3 (MaxPooling2 |(None, 7, 7, 64)    |      0    |     
reshape_1 (Reshape)      |   (None, 3136)        |      0      |   
flatten_1 (Flatten)       |   (None, 3136)      |        0        | 
dense_2 (Dense)       |       (None, 1024)        |      3212288   |
dropout_1 (Dropout)     |     (None, 1024)       |       0         |
dense_3 (Dense)         |     (None, 10)           |     10250     |
-------------------------------------------------------------------

Total params: 3,274,634
Trainable params: 3,274,634

#### Steps to reproduce our results:

1. Follow the GCP instructions[https://www.kubeflow.org/docs/gke/deploy/] to deploy Kubeflow 

2. Requirements

    * Kubeflow 1.0 on Kubernetes Engine. See the guide[https://www.kubeflow.org/docs/gke/deploy]
    * Kubeflow Cluster 
    * Launch a Notebook instane on Kubeflow and run the following code using bash termnial

3. Training a CNN on MNIST, CIFAR10 data-sets on Kubeflow 

`pip install msrestazure` 

You must be running Kubeflow 1.0 on Kubernetes Engine (GKE) with Cloud Identity-Aware Proxy (Cloud IAP). See the guide to deploying Kubeflow on GCP[https://www.kubeflow.org/docs/gke/deploy/]

Running KubeFlow for MNIST data on 2 layer CNN
`python3 kubeflow_main_script.py --tf-model-script=kubeflow_mnist_model.py` 

Running KubeFlow for CIFAR10 data on 2 layer CNN
`python3 kubeflow_main_script.py --tf-model-script=kubeflow_cifar10_model.py` 

4. Training a CNN on MNIST data on Fiber 

`pip install fiber` 

`python3 fiber_cifar10.py`

`python3 fiber_mnist.py`

5. Use the GCP metrics explorer to track metrics[(https://cloud.google.com/monitoring/charts/metrics-explorer]

### Results:

|   dataset|  architecture |  Distribution engine (fiber / KubeFlow) | Training time(secs)  | CPU Utilization (%)   |  Peak disk read bytes  |  Peak disk write bytes  |
|---|---|---|---|---|---|---|
|  MNIST    | 2 layer CNN  | KubeFlow  |  2  | 34.64 | 358 KiB  | 64.662 MiB  |
|  MNIST    | 2 layer CNN  |   Fiber   |  35 | 85.03 | 488 KiB  | 1.262  MiB  |
| CIFAR -10 | 2 layer CNN  | KubeFlow  | 8.6 | 56.37 | 2.00 KiB | 4.186  MiB  |
| CIFAR -10 | 2 layer CNN  |   Fiber   |  48 | 75.65 | 28.00 KiB| 38.94  MiB  |


### Conclusion:

In  conclusion,  this  study  achieved  its  primary  goal  of effectively comparing the platforms ’Fiber’ and ’Kubeflow’ for potential users. It showed that while Kubeflow exceeds Fiber with respect to training speed and CPU utilization efficiency,Fiber  has  the  edge  when  it  comes  to  throughput.  We  also revealed that Kubeflow has hidden costs involved but its price is  worth  paying  for  the  usability  it  brings.  We  still  think that  Fiber  can  be  a  promising  contender  in  the  future  if  the developers pay heed to the time and CPU metrics as well as work  on  making  it  seamlessly  integrable  with  all  the  major ML python frameworks.


### References & Citations :
1. Fiber : ​https://uber.github.io/fiber/
2. TF Traning : https://www.kubeflow.org/docs/components/training/tftraining/
3. Visit the Kubeflow docs(https://www.kubeflow.org/docs/gke/) for more information about running Kubeflow on GCP
4. MNIST model on tensorflow 1x : https://github.com/tensorflow/docs/blob/master/site/en/r1/tutorials/estimators/cnn.ipynb
5. Reference for kubeflow : https://github.com/kubeflow/examples/ 

-- git status 
-- git add .
-- git commit -m "changes to the readmefile done"
-- git push 