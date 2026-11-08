FROM amazonlinux:2

# Install system dependencies
RUN yum update -y && \
    yum groupinstall -y "Development Tools" && \
    yum install -y gcc wget make zip tar \
    bzip2-devel libffi-devel xz-devel \
    readline-devel sqlite-devel zlib-devel \
    openssl11 openssl11-devel \
    postgresql-devel

# Set env to use OpenSSL 1.1 (Amazon Linux 2 default is OpenSSL 1.0)
ENV LD_LIBRARY_PATH=/usr/lib64/openssl11
ENV CFLAGS="-I/usr/include/openssl11"
ENV LDFLAGS="-L/usr/lib64/openssl11"

# Download and build Python from source
WORKDIR /opt
RUN wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz && \
    tar xzf Python-3.10.14.tgz && \
    cd Python-3.10.14 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall

# Use Python 3.10 and pip
RUN ln -sf /usr/local/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/local/bin/pip3.10 /usr/bin/pip

# Build the wheel
WORKDIR /build
RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --wheel-dir=/build psycopg2-binary
