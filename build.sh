#!/bin/bash
VERSION=$(grep '^version = ' pyproject.toml | cut -d '"' -f2)
export ECHO_LATEST_TAG="$VERSION"
docker buildx create --use --name python3-builder --node python3-builder0 --driver docker-container --driver-opt image=moby/buildkit:v0.10.6
docker buildx build -f docker/Dockerfile --platform linux/amd64,linux/arm64 --tag ifelsedotone/echo:latest --tag ifelsedotone/echo:${ECHO_LATEST_TAG} . --push