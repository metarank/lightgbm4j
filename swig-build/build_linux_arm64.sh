#!/bin/bash

docker buildx build --platform arm64 -t lgbm:latest -f "Dockerfile.arm64" .