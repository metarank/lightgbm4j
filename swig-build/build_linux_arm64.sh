#!/bin/bash
docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
docker buildx build --platform arm64 -t lgbm:latest -f "Dockerfile.arm64" .