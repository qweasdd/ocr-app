FROM debian:sid

ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    python3 python3-pip python3-dev python3-setuptools \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev libgtk2.0-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*

RUN pip3 install --no-cache-dir deep-pipe matplotlib opencv-python albumentations pandas trafaret trafaret-config==2.0.2

RUN pip3 install --no-cache-dir -U catalyst

ENV PYTHONPATH $PYTHONPATH:/workdir/src

COPY . /workdir

WORKDIR /workdir
