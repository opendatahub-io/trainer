# Build the manager binary
# Build context is the repository root so the nested module can resolve its
# local replace of github.com/kubeflow/trainer/v2 (../../ from this directory).
FROM registry.access.redhat.com/ubi9/go-toolset:1.26 as builder

WORKDIR /workspace
# Copy the local trainer module and operator module manifests used by go.mod.
COPY go.mod go.sum ./
COPY pkg/apis/ ./pkg/apis/
COPY rhoai/operator/go.mod rhoai/operator/go.sum ./rhoai/operator/

WORKDIR /workspace/rhoai/operator
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
# Retry: proxy.golang.org occasionally returns HTTP/2 INTERNAL_ERROR.
RUN go mod download || (sleep 5 && go mod download) || (sleep 10 && go mod download)

# Copy the go source
COPY rhoai/operator/ ./

# Build
USER root
RUN CGO_ENABLED=1 GOEXPERIMENT=strictfipsruntime GOOS=linux go build -a -o manager cmd/main.go

# Runtime
FROM registry.access.redhat.com/ubi9/ubi-minimal:9.8

WORKDIR /
COPY --from=builder /workspace/rhoai/operator/manager .
COPY manifests/base/ /opt/manifests-template/base/
COPY manifests/rhoai/ /opt/manifests-template/rhoai/
COPY manifests/component_metadata.yaml /opt/manifests-template/
COPY rhoai/operator/manifests/runtimes/ /opt/runtimes-template/
COPY rhoai/operator/manifests/imagestreams/ /opt/imagestreams-template/
USER 65532:65532

ENTRYPOINT ["/manager"]
