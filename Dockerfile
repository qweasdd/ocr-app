FROM  ubuntu:18.04

ENV LANG C.UTF-8

RUN apt-get update &&\
    apt-get -y install build-essential yasm nasm cmake unzip git wget \
    sysstat libtcmalloc-minimal4 pkgconf autoconf libtool flex bison \
    python3 python3-pip python3-dev python3-setuptools \
    libsm6 libxext6 libxrender1 libssl-dev libx264-dev &&\
    ln -s /usr/bin/python3 /usr/bin/python &&\
    ln -s /usr/bin/pip3 /usr/bin/pip &&\
    apt-get clean &&\
    apt-get autoremove &&\
    rm -rf /var/lib/apt/lists/* &&\
    rm -rf /var/cache/apt/archives/*


RUN apt-get update
RUN git clone --depth 1 -b release/4.2 --single-branch https://github.com/FFmpeg/FFmpeg.git &&\
    cd FFmpeg &&\
    mkdir ffmpeg_build && cd ffmpeg_build &&\
    ../configure \
    --enable-shared \
    --disable-static \
    --disable-doc \
    --extra-cflags=-I/usr/local/cuda/include \
    --extra-ldflags=-L/usr/local/cuda/lib64 \
    --enable-nonfree \
    --enable-gpl \
    --enable-openssl \
    --enable-libx264 \
    --extra-libs=-lpthread \
    --nvccflags="-gencode arch=compute_75,code=sm_75" &&\
    make -j$(nproc) && make install && ldconfig &&\
    cd ../.. && rm -rf FFmpeg

RUN pip3 install --no-cache-dir deep-pipe matplotlib opencv-python albumentations pandas trafaret trafaret-config==2.0.2

RUN pip3 install --no-cache-dir -U catalyst


ENV PYTHONPATH $PYTHONPATH:/workdir/src


COPY . /workdir

WORKDIR /workdir

ENTRYPOINT ["python"]
