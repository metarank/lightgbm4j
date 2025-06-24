#!/bin/bash
docker buildx build --platform amd64 -t lgbm:latest -f "Dockerfile.amd64" .