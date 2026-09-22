# syntax=docker/dockerfile:1
# Linux native build for lightgbm4j. glibc 2.31 (ubuntu focal) is the compatibility
# baseline of the shipped .so files. Driven by native/linux.sh; works on amd64 and
# arm64 hosts without qemu (the toolchain is picked by `uname -m`).
FROM ubuntu:focal-20240427 AS toolchain
ARG DEBIAN_FRONTEND=noninteractive
ARG CMAKE_VERSION=3.31.6
# SWIG is pinned: it decides the generated Java in src/main/java/com/microsoft/ml/lightgbm
ARG SWIG_VERSION=4.2.1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl build-essential g++ openjdk-11-jdk \
      autoconf automake libtool bison libpcre2-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-$(uname -m).tar.gz" \
    | tar -xz --strip-components=1 -C /usr/local
RUN curl -fsSL "https://github.com/swig/swig/archive/refs/tags/v${SWIG_VERSION}.tar.gz" | tar -xz -C /tmp \
    && cd "/tmp/swig-${SWIG_VERSION}" && ./autogen.sh && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" && make install && rm -rf /tmp/swig-*

FROM toolchain AS build
WORKDIR /src
COPY native/lightgbm .
# JAVA_HOME differs by arch (java-11-openjdk-amd64 / -arm64), so derive it from javac
RUN export JAVA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v javac)")")")" \
    && cmake -B build -S . -DUSE_SWIG=ON -DBUILD_CLI=OFF -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j"$(nproc)"
# upstream's POST_BUILD step stages both libs under build/com/microsoft/ml/lightgbm/linux/x86_64
# (path is hard-coded x86_64 upstream, even on aarch64)
RUN ARCH="$(uname -m)"; OUT="/dist/resources/lightgbm4j/linux/$ARCH"; \
    mkdir -p "$OUT" /dist/java/com/microsoft/ml/lightgbm \
    && for lib in lib_lightgbm lib_lightgbm_swig; do \
         cp "build/com/microsoft/ml/lightgbm/linux/x86_64/$lib.so" "$OUT/$lib.so" \
         && md5sum "$OUT/$lib.so" | cut -d' ' -f1 | tr -d '\n' > "$OUT/$lib.so.md5"; \
       done \
    && cp build/java/*.java /dist/java/com/microsoft/ml/lightgbm/ \
    && ls -lR /dist

FROM scratch AS dist
COPY --from=build /dist /
