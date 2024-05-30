#!/bin/bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker buildx build --platform amd64 -t lgbm:latest -f "Dockerfile.amd64" .