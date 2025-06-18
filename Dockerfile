FROM ubuntu:22.04

LABEL description="Ubuntu 22.04 with Acts, Pythia8, Fatras, Python bindings, JupyterLab"
LABEL maintainer="Eugene Shalugin <eugene.shalugin@ru.nl>"
LABEL version="6"

ENV DEBIAN_FRONTEND=noninteractive
ENV PREFIX=/usr/local
ENV PYTHIA8_DIR=/usr/local
ENV CMAKE_PREFIX_PATH=/usr/local
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=/usr/local/lib:${PYTHONPATH}
ENV PATH=/opt/cmake/bin:$PATH

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential g++ curl wget git ninja-build cmake pkg-config ccache \
    python3 python3-dev python3-pip python3-venv rsync \
    libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-test-dev \
    libeigen3-dev libexpat-dev libftgl-dev libgl2ps-dev libglew-dev libgsl-dev \
    liblz4-dev liblzma-dev libpcre3-dev libsqlite3-dev libtbb-dev libx11-dev \
    libxext-dev libxft-dev libxpm-dev libxerces-c-dev libxxhash-dev libzstd-dev \
    zlib1g-dev libxml2-dev libssl-dev libglu1-mesa-dev libmysqlclient-dev \
    libfftw3-dev libcfitsio-dev graphviz-dev libavahi-compat-libdnssd-dev \
    libldap2-dev libkrb5-dev libtiff-dev libpng-dev libz-dev libbz2-dev \
    libblas-dev liblapack-dev libxmu-dev libxi-dev libedit-dev libncurses-dev \
    libreadline-dev ca-certificates && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install CMake 3.26.4 to avoid ROOT build issues
RUN wget https://github.com/Kitware/CMake/releases/download/v3.26.4/cmake-3.26.4-linux-x86_64.tar.gz && \
    tar -xzf cmake-3.26.4-linux-x86_64.tar.gz && \
    mv cmake-3.26.4-linux-x86_64 /opt/cmake && \
    rm cmake-3.26.4-linux-x86_64.tar.gz

# JSON helper variables
ENV GET curl --location --silent --create-dirs
ENV UNPACK_TO_SRC tar -xz --strip-components=1 --directory src

# Install HepMC3
RUN mkdir src && \
    ${GET} https://gitlab.cern.ch/hepmc/HepMC3/-/archive/3.2.6/HepMC3-3.2.6.tar.gz | ${UNPACK_TO_SRC} && \
    cmake -B build -S src -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -DCMAKE_CXX_STANDARD=20 \
      -DHEPMC3_BUILD_STATIC_LIBS=OFF \
      -DHEPMC3_ENABLE_PYTHON=OFF \
      -DHEPMC3_ENABLE_ROOTIO=OFF \
      -DHEPMC3_ENABLE_SEARCH=OFF && \
    cmake --build build -- install && \
    rm -rf build src

# Install nlohmann_json
RUN mkdir src && \
    ${GET} https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz | ${UNPACK_TO_SRC} && \
    cmake -B build -S src -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DJSON_BuildTests=OFF && \
    cmake --build build -- install && \
    rm -rf build src

# Install ROOT (v6.30.06)
RUN mkdir src && \
    ${GET} https://root.cern/download/root_v6.30.06.source.tar.gz | ${UNPACK_TO_SRC} && \
    sed -i 's/COMMENT .*//g' src/interpreter/llvm-project/clang/lib/Tooling/CMakeLists.txt && \
    cmake -B build -S src -GNinja \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -Dcxxstd=20 \
      -DCMAKE_INSTALL_PREFIX=${PREFIX} \
      -Dfail-on-missing=ON \
      -Dgminimal=ON \
      -Dgdml=ON \
      -Dopengl=ON \
      -Dpyroot=OFF && \
    cmake --build build -- install && \
    rm -rf build src

## Install Pythia8
RUN mkdir src && \
    curl -Ls https://pythia.org/download/pythia83/pythia8311.tgz | tar -xz --strip-components=1 -C src && \
    cd src && \
    ./configure --enable-shared --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd .. && rm -rf src



## Install ACTS
RUN wget https://github.com/acts-project/acts/archive/refs/tags/v41.1.0.tar.gz && \
    tar -xvzf v41.1.0.tar.gz && \
    rm v41.1.0.tar.gz && \
    cd acts-41.1.0 && \
    python3 -m venv venv_acts && \
    . venv_acts/bin/activate && \
    pip install --upgrade pip && \
    cmake -B build -S . \
      -DACTS_BUILD_EXAMPLES_PYTHON_BINDINGS=ON \
      -DACTS_BUILD_FATRAS=ON \
      -DACTS_BUILD_EXAMPLES_PYTHIA8=ON \
      -DACTS_BUILD_EXAMPLES_ROOT=ON \
      -DCMAKE_POLICY_VERSION_MINIMUM=3.26.4 \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_CXX_STANDARD=20 \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build && \
    cd ..


RUN . /app/acts-41.1.0/venv_acts/bin/activate && pip install jupyterlab

# Expose Jupyter Lab
EXPOSE 8000
CMD bash -c "source /app/acts-41.1.0/venv_acts/bin/activate && source /app/acts-41.1.0/build/python/setup.sh && jupyter lab --ip=0.0.0.0 --port=8000 --no-browser --allow-root"

