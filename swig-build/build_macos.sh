#!/bin/sh

# brew install
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
(echo; echo 'eval "$(/opt/homebrew/bin/brew shellenv)"') >> /Users/m1/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

# install deps
brew install libomp git openjdk@11 cmake swig htop mc

sudo ln -sfn /opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk-11.jdk
echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc

export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/openjdk@11/include -I/opt/homebrew/opt/libomp/include"
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@11/11.0.27/libexec/openjdk.jdk/Contents/Home
export CC=gcc CXX=g++

# checkout

git clone --recursive --depth 1 --branch v4.4.0 https://github.com/microsoft/LightGBM
cd LightGBM 
cmake -B build -S . -DUSE_SWIG=ON
cmake --build build -j6
