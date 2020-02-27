FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt-get update && apt-get install -y python3.6 python3.6-dev

RUN apt-get install -y git

RUN git clone https://github.com/zju3dv/mvpose.git

RUN apt-get install -y python3-pip && python3.6 -m pip install --upgrade pip

RUN cd mvpose && python3.6 -m pip install -r requirements.txt

RUN apt-get remove -y python3 && rm /usr/bin/python3 && ln -s /usr/bin/python3.6 /usr/bin/python3

# compile backend
RUN cd mvpose/backend/tf_cpn/lib/ && make
RUN cd mvpose/backend/tf_cpn/lib/lib_kernel/lib_nms/ && ./compile.sh

RUN sed -i -- 's=cuda/include/cuda.h=cuda.h=g' /usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/cuda_launch_config.h
RUN sed -i -- 's=cuda/include/cuda.h=cuda.h=g' /usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/cuda_device_functions.h
RUN sed -i -- 's=cuda/include/cuda_fp16.h=cuda_fp16.h=g' /usr/local/lib/python3.6/dist-packages/tensorflow/include/tensorflow/core/util/cuda_kernel_helper.h

# run the following on entry:
#RUN cd mvpose/backend/light_head_rcnn/lib/ && bash make.sh
RUN sed -i -- 's=from Cython.Distutils import build_ext=from Cython.Distutils import build_ext\nimport numpy=g' /mvpose/src/m_lib/setup.py
RUN sed -i -- 's~\["pictorial.pyx"\],~["pictorial.pyx"],\ninclude_dirs=[numpy.get_include()],~g' /mvpose/src/m_lib/setup.py
RUN cd mvpose/src/m_lib && python3 setup.py build_ext --inplace

# download and extract data
RUN python3.6 -m pip install gdown

RUN mkdir -p /mvpose/backend/light_head_rcnn/output/model_dump
RUN cd mvpose/backend/light_head_rcnn/output && gdown -O file.tar.gz https://drive.google.com/uc?id=1klpM_DEIn2Ln4ZN-xWHdvwp40dYpQ05b && tar -xvf file.tar.gz
RUN rm mvpose/backend/light_head_rcnn/output/file.tar.gz

RUN mkdir -p /mvpose/backend/tf_cpn/log/model_dump
RUN cd /mvpose/backend/tf_cpn/log && gdown -O file.tar.gz https://drive.google.com/uc?id=1DJF4p-SC_PokGtt7TbCVPgo-EWRQYhGi && tar -xvf file.tar.gz
RUN rm /mvpose/backend/tf_cpn/log/file.tar.gz

RUN mkdir -p /mvpose/backend/CamStyle/logs
RUN cd /mvpose/backend/CamStyle && gdown -O file.tar.gz https://drive.google.com/uc?id=1FRAu6sr0Bd39ZliCscum69mwuZ1j502b && tar -xvf file.tar.gz
RUN rm /mvpose/backend/CamStyle/file.tar.gz

# load dataset
RUN mkdir /mvpose/datasets
RUN mkdir /mvpose/result
RUN apt-get install -y wget
RUN cd /mvpose/datasets && wget http://campar.cs.tum.edu/files/belagian/multihuman/CampusSeq1.tar.bz2
RUN cd /mvpose/datasets && tar -xvf CampusSeq1.tar.bz2
RUN rm /mvpose/datasets/CampusSeq1.tar.bz2

# load pre trained camera_parameter pickle
RUN cd /mvpose/datasets/CampusSeq1 && gdown https://drive.google.com/uc?id=1BvIyB53Jb_asZ2gEoIRh8gYUvHxPDcPA

# install other libs for training
RUN apt-get update && apt-get install -y libsm6 libxext6 libxrender-dev python3.6-tk
RUN python3.6 -m pip install ipython sklearn Pillow==6.1

# Entrypoint
COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
RUN ln -s /usr/local/bin/entrypoint.sh /
ENTRYPOINT ["entrypoint.sh"]
CMD ["bash"]
