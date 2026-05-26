# Build the manager binary
FROM registry.access.redhat.com/ubi9/go-toolset:1.26 as builder

WORKDIR /workspace
# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the go source
COPY . .

# Build
USER root
RUN CGO_ENABLED=1 GOEXPERIMENT=strictfipsruntime GOOS=linux go build -a -o manager cmd/main.go

# Collect trainer manifests
RUN bash hack/get_trainer_manifests.sh

# Runtime
FROM registry.access.redhat.com/ubi9/ubi-minimal:latest

WORKDIR /
COPY --from=builder /workspace/manager .
COPY --from=builder /workspace/opt/manifests/ /opt/manifests-template/
COPY manifests/runtimes/ /opt/runtimes-template/
COPY manifests/imagestreams/ /opt/imagestreams-template/
USER 65532:65532

ENTRYPOINT ["/manager"]
