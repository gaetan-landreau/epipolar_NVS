# EpipolarNVS 

This repository contains the whole code structure related to the EpipolarNVS contribution. Our main work focuses on integrating 3D prior information within a deep neural network that is devoted to single-image novel view synthesis. 

Both the datasets and weight files will be made public and accessible to all (through a Gdrive link) subject to acceptance of our contribution. 

**Code Overview - Installation - Basic usage**

The whole code has been developped through a Docker container that support NVIDIA GPUs and Tensorflow 2.7. We both provide the *Dockerfile* that we used to build our image as well as a *docker-compose.yaml* to launch a basic container from it with a JupyterLab session. One would make sure by itsself that the different needed volumes are properly mounted on the container. 

The Docker image can be built in the root project folder through: 
```bash
docker build -t name_of_your_image:latest . 
```

From there, the container can easely be launch with: 

```bash
docker-compose up -d main #main is the service name.
```

From a general perspective, both training and inference can easely be run through the corresponding *train.py* and *test.py* files. All the different hyperparameters (encoding strategy, dataset used, batch size, learning rate etc)  are stored within the .YAML file *params.yaml* . 

One can therefore easely launch a training with : 
```python 
python train.py
```

or perform inference with: 
```python 
python test.py
```

