FROM tensorflow/tensorflow:1.6.0-py3
RUN mkdir /Software
WORKDIR /Software

# Install Aleph dependencies
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y cmake git python3-dev
RUN pip3 install pytest
RUN git clone https://github.com/pybind/pybind11.git pybind11
RUN mkdir pybind11/build
WORKDIR pybind11/build
RUN cmake ..
#RUN make -j 4
RUN make -j 4 install

# Install Aleph python module
RUN apt-get install -y libboost-dev libboost-regex-dev 
RUN git clone https://github.com/Submanifold/Aleph.git Aleph
RUN mkdir Aleph/build
WORKDIR Aleph/build
RUN git checkout 8c88506
RUN cmake ..
RUN make aleph
RUN pip3 install bindings/python/aleph

# Install python dependencies
RUN pip3 install sacred matplotlib seaborn pandas

RUN mkdir /Neuralpersistence
WORKDIR /Neuralpersistence
ADD . /Neuralpersistence
