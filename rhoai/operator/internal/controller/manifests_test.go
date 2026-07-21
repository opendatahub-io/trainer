package controller

import (
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
