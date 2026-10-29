package controller

import (
	"context"

	"os"
	"path/filepath"
	"testing"

	. "github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"

	configv1 "github.com/openshift/api/config/v1"
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

// operandTestConfig reads the real operand configuration file so the
// injection tests exercise the shipped content.
func operandTestConfig(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "..", "manifests", "trainer", "rhoai", "controller_manager_config.yaml"))
	if err != nil {
		t.Fatalf("reading operand config: %v", err)
	}
	return string(raw)
}

func mkOperandConfigMap(t *testing.T, config string) unstructured.Unstructured {
	cm := unstructured.Unstructured{}
	cm.SetKind("ConfigMap")
	cm.SetAPIVersion("v1")
	// The kustomize configMapGenerator appends a content hash to the name.
	cm.SetName("kubeflow-trainer-config-7d9h8t6b4f")
	err := unstructured.SetNestedField(cm.Object, config, "data", operandConfigKey)
	if err != nil {
		t.Fatalf("setting config data: %v", err)
	}
	return cm
}

func mkOperandDeployment() unstructured.Unstructured {
	dep := unstructured.Unstructured{}
	dep.SetKind("Deployment")
	dep.SetAPIVersion("apps/v1")
	dep.SetName(operandDeploymentName)
	return dep
}

func operandTLSBlock(t *testing.T, g Gomega, cm unstructured.Unstructured) map[string]any {
	t.Helper()
	raw, found, err := unstructured.NestedString(cm.Object, "data", operandConfigKey)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(found).To(BeTrue())

	var cfg map[string]any
	g.Expect(yaml.Unmarshal([]byte(raw), &cfg)).To(Succeed())
	return cfg["tls"].(map[string]any)
}

func TestApplyOperandTLSOptionsInjectsConfigAndAnnotation(t *testing.T) {
	g := NewWithT(t)

	config := operandTestConfig(t)
	items := []unstructured.Unstructured{mkOperandConfigMap(t, config), mkOperandDeployment()}

	err := applyOperandTLSOptions(context.Background(), items, *configv1.TLSProfiles[configv1.TLSProfileIntermediateType])
	g.Expect(err).NotTo(HaveOccurred())

	tlsBlock := operandTLSBlock(t, g, items[0])
	g.Expect(tlsBlock["minVersion"]).To(Equal("1.2"))
	g.Expect(tlsBlock["cipherSuites"]).To(ContainElement("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"))

	// Existing configuration keys are preserved.
	raw, _, err := unstructured.NestedString(items[0].Object, "data", operandConfigKey)
	g.Expect(err).NotTo(HaveOccurred())
	var cfg map[string]any
	g.Expect(yaml.Unmarshal([]byte(raw), &cfg)).To(Succeed())
	g.Expect(cfg["featureGates"]).To(Equal(map[string]any{"TrainJobDynamicResourceAllocation": true}))

	// The operand deployment carries the config hash annotation.
	annotations, found, err := unstructured.NestedStringMap(items[1].Object, "spec", "template", "metadata", "annotations")
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(found).To(BeTrue())
	g.Expect(annotations).To(HaveKey(operandTLSConfigHashAnnotation))
}

func TestApplyOperandTLSOptionsEmptyOperandConfig(t *testing.T) {
	g := NewWithT(t)

	// A comment-only operand config parses to a nil map; injection must
	// not panic, must still add the tls block, and must still annotate.
	items := []unstructured.Unstructured{mkOperandConfigMap(t, "# upstream sync shipped an empty config\n"), mkOperandDeployment()}
	g.Expect(applyOperandTLSOptions(context.Background(), items, *configv1.TLSProfiles[configv1.TLSProfileIntermediateType])).To(Succeed())

	tlsBlock := operandTLSBlock(t, g, items[0])
	g.Expect(tlsBlock["minVersion"]).To(Equal("1.2"))

	annotations, found, err := unstructured.NestedStringMap(items[1].Object, "spec", "template", "metadata", "annotations")
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(found).To(BeTrue())
	g.Expect(annotations).To(HaveKey(operandTLSConfigHashAnnotation))
}

func TestApplyOperandTLSOptionsHashRollsWithProfile(t *testing.T) {
	g := NewWithT(t)

	config := operandTestConfig(t)

	first := []unstructured.Unstructured{mkOperandConfigMap(t, config), mkOperandDeployment()}
	g.Expect(applyOperandTLSOptions(context.Background(), first, *configv1.TLSProfiles[configv1.TLSProfileIntermediateType])).To(Succeed())
	firstHash, _, err := unstructured.NestedString(first[1].Object, "spec", "template", "metadata", "annotations", operandTLSConfigHashAnnotation)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(firstHash).NotTo(BeEmpty())

	second := []unstructured.Unstructured{mkOperandConfigMap(t, config), mkOperandDeployment()}
	g.Expect(applyOperandTLSOptions(context.Background(), second, *configv1.TLSProfiles[configv1.TLSProfileModernType])).To(Succeed())
	secondHash, _, err := unstructured.NestedString(second[1].Object, "spec", "template", "metadata", "annotations", operandTLSConfigHashAnnotation)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(secondHash).NotTo(BeEmpty())

	g.Expect(firstHash).NotTo(Equal(secondHash))
}

func TestApplyOperandTLSOptionsNoConfigMapNoAnnotation(t *testing.T) {
	g := NewWithT(t)

	items := []unstructured.Unstructured{mkOperandDeployment()}
	g.Expect(applyOperandTLSOptions(context.Background(), items, *configv1.TLSProfiles[configv1.TLSProfileIntermediateType])).To(Succeed())

	_, found, err := unstructured.NestedString(items[0].Object, "spec", "template", "metadata", "annotations", operandTLSConfigHashAnnotation)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(found).To(BeFalse())
}

func TestApplyOperandTLSOptionsGroupOnlyChangeRollsHash(t *testing.T) {
	g := NewWithT(t)

	config := operandTestConfig(t)

	withX25519 := configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Groups:        []configv1.TLSGroup{configv1.TLSGroupX25519},
	}
	withMLKEM := configv1.TLSProfileSpec{
		MinTLSVersion: configv1.VersionTLS12,
		Groups:        []configv1.TLSGroup{configv1.TLSGroupX25519MLKEM768},
	}

	hashFor := func(profile configv1.TLSProfileSpec) string {
		items := []unstructured.Unstructured{mkOperandConfigMap(t, config), mkOperandDeployment()}
		g.Expect(applyOperandTLSOptions(context.Background(), items, profile)).To(Succeed())
		hash, _, err := unstructured.NestedString(items[1].Object, "spec", "template", "metadata", "annotations", operandTLSConfigHashAnnotation)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(hash).NotTo(BeEmpty())
		return hash
	}

	// A group-only change (same version, same ciphers) must change the
	// rendered configuration hash so the operand pods roll.
	g.Expect(hashFor(withX25519)).NotTo(Equal(hashFor(withMLKEM)))

	// The curve preferences must also land in the rendered config itself.
	items := []unstructured.Unstructured{mkOperandConfigMap(t, config), mkOperandDeployment()}
	g.Expect(applyOperandTLSOptions(context.Background(), items, withMLKEM)).To(Succeed())
	raw, found, err := unstructured.NestedString(items[0].Object, "data", operandConfigKey)
	g.Expect(err).NotTo(HaveOccurred())
	g.Expect(found).To(BeTrue())
	g.Expect(raw).To(ContainSubstring("curvePreferences:"))
	g.Expect(raw).To(ContainSubstring("- 4588"))
}
