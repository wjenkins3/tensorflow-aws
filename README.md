# TensorFlow on Amazon EC2

This repository contains basic setup information for using TensorFlow on Amazon EC2 GPU instances.<br/>
<br/>
MNIST Convolution.ipynb provides an example of a Convolutional Neural Network for
classifying MNIST data. It provides a partial TensorFlow implementation of the CNN from
Michael Nielsen's book [Neural Networks and Deep
Learning](http://neuralnetworksanddeeplearning.com/).

## Installing TensorFlow on EC2

This is the process I followed to install TensorFlow on an AWS EC2 GPU instance. All that follows is based on the following posts:<br/>
<https://github.com/Avsecz/aws-tensorflow-setup>
<br/>
<http://eatcodeplay.com/installing-gpu-enabled-tensorflow-with-python-3-4-in-ec2/>
<br/>

### The following is installed on an instance running Ubuntu LTS 14.04 AMI.

- Required Linux packages
- Anaconda for Python 2.7
- CUDA 7.5
- cuDNN 4.0
- TensorFlow 0.10

### Install

Some essentials

```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install -y build-essential git python-pip swig zip unzip wget python-dev
```

**Anaconda**

```
wget https://repo.continuum.io/archive/Anaconda2-4.1.1-Linux-x86_64.sh
bash Anaconda2-4.1.1-Linux-x86_64.sh -p ~/bin/anaconda2
rm Anaconda2-4.1.1-Linux-x86_64.sh
source .bash_rc
```

Setup **Jupyter Notebook**

```
ipython profile create nbserver
ipython
```

Enter a password when prompted

```
>> from IPython.lib import passwd
>> passwd()
Enter password:
Verify password:
>> !mkdir ~/bin/certificates
>> cd ~/bin/certificates
>> !openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout mycert.pem -out mycert.pem
>> quit()
```

Generate **jupyter_notebook_config.py**

```
jupyter notebook --generate-config
cd ~/.jupyter
```

With the editor of your choice, edit **jupyter_notebook_config.py**. Rather than searching for
and uncommenting each line, it is easier to just copy/paste the following to the top of
the file.
   ```
   c = get_config()
   c.IPKernelApp.pylab = 'inline'
   c.NotebookApp.certfile = u'/home/ubuntu/bin/certificates/'
   c.NotebookApp.ip = '*'
   c.NotebookApp.open_browser = False
   c.NotebookApp.password = 'u:sha1:<hashed password here>'
   c.NotebookApp.port = 8888
   ```

```
# To start server
jupyter notebook --profile=nbserver
```
Jupyter Notebook setup is complete. Try going to http://INSTANCE_PUBLIC_IP_ADDRESS:8888 or http://INSTANCE_PUBLIC_DNS_ADDRESS:8888 in your browser.<br/>
<br/>
**CUDA** and **CUDNN**
```
sudo apt-get -y install linux-headers-$(uname -r) linux-image-extra-`uname -r`
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_7.5-18_amd64.deb
rm cuda-repo-ubuntu1404_7.5-18_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda

CUDNN_FILE=cudnn-7.0-linux-x64-v4.0-prod.tgz
wget http://developer.download.nvidia.com/compute/redist/cudnn/v4/${CUDNN_FILE}
tar xvzf ${CUDNN_FILE}
rm ${CUDNN_FILE}
sudo cp cuda/include/cudnn.h /usr/local/cuda/include # move library files to /usr/local/cuda
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
rm -rf cuda

# set the appropriate library path
echo 'export CUDA_HOME=/usr/local/cuda
export CUDA_ROOT=/usr/local/cuda
export PATH=$PATH:$CUDA_ROOT/bin:$HOME/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_ROOT/lib64
' >> ~/.bashrc

sudo reboot
```

After the reboot, you are ready to install **TensorFlow**

```
# Ubuntu/Linux 64-bit, GPU enabled, Python 2.7
# Requires CUDA toolkit 7.5 and CuDNN v4. For other versions, see "Install from sources" below.
export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.10.0rc0-cp27-none-linux_x86_64.whl

pip install --upgrade --ignore-installed $TF_BINARY_URL

# Test installation
python
>> import tensorflow as tf
>> quit()
```

## Notes

The model in the IPython notebook example does not achieve state-of-the-art accuracy on
the MNIST data. The code does, however, provide a great starting point for completing
Nielsen's implementation; the MNIST dataset and code samples for his book are located
[here](https://github.com/mnielsen/neural-networks-and-deep-learning). <br/>
<br/>
Feel free to report any issues you find with the process. You may also want to check out the links above; I found the posts very helpful.