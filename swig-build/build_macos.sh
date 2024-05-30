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
export JAVA_HOME=/opt/homebrew/Cellar/openjdk@11/11.0.21/libexec/openjdk.jdk/Contents/Home
export CC=gcc-13 CXX=g++-13

# checkout

git clone --recursive https://github.com/microsoft/LightGBM
cmake -DUSE_SWIG=ON -DAPPLE_OUTPUT_DYLIB=ON ..