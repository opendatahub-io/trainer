package controller

import (
	"path/filepath"
	"testing"

	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

func TestFilterConfigMaps(t *testing.T) {
	g := NewWithT(t)

	mkObj := func(kind string) unstructured.Unstructured {
		o := unstructured.Unstructured{}
		o.SetKind(kind)
		return o
	}

	items := []unstructured.Unstructured{mkObj("Deployment"), mkObj("ConfigMap"), mkObj("Service")}

	filtered := filterConfigMaps(items)
	g.Expect(filtered).To(HaveLen(2))
	g.Expect(filtered[0].GetKind()).To(Equal("Deployment"))
	g.Expect(filtered[1].GetKind()).To(Equal("Service"))
}

func TestRenderDefaultOverlayPlacesPrometheusRuleInOperatorNamespace(t *testing.T) {
	g := NewWithT(t)

	rendered, err := renderOverlay(filepath.Join("..", "..", "config", "default"), "")
	g.Expect(err).NotTo(HaveOccurred())

	for _, resource := range rendered {
		if resource.GetKind() == "PrometheusRule" && resource.GetName() == "trainer-operator-kubeflow-trainer-telemetry" {
			g.Expect(resource.GetNamespace()).To(Equal("trainer-operator-system"))
			return
		}
	}

	t.Fatal("Trainer PrometheusRule was not rendered")
}
