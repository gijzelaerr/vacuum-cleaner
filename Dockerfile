FROM tensorflow/tensorflow:1.11.0-gpu-py3
RUN apt-get update && apt-get -y install python3-astropy python3-scipy
ADD . /code
WORKDIR /code
RUN pip install .
