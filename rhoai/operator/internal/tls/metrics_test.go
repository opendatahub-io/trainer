/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tls

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/onsi/gomega"
)

func TestMetricsServerOptions(t *testing.T) {
	tlsOpts := []func(*tls.Config){func(c *tls.Config) { c.MinVersion = tls.VersionTLS12 }}

	t.Run("HTTP skips auth filter and cert dir", func(t *testing.T) {
		g := NewWithT(t)
		opts := MetricsServerOptions(":8080", false, "", tlsOpts)

		g.Expect(opts.BindAddress).Should(Equal(":8080"))
		g.Expect(opts.SecureServing).Should(BeFalse())
		g.Expect(opts.FilterProvider).Should(BeNil())
		g.Expect(opts.CertDir).Should(BeEmpty())
		g.Expect(opts.TLSOpts).Should(HaveLen(1))
	})

	t.Run("HTTPS enables auth filter without requiring a cert dir", func(t *testing.T) {
		g := NewWithT(t)
		opts := MetricsServerOptions(":8443", true, "", tlsOpts)

		g.Expect(opts.SecureServing).Should(BeTrue())
		g.Expect(opts.FilterProvider).ShouldNot(BeNil())
		g.Expect(opts.CertDir).Should(BeEmpty())
		g.Expect(opts.TLSOpts).Should(HaveLen(1))
	})

	t.Run("HTTPS with cert path sets CertDir and loads certs on handshake", func(t *testing.T) {
		g := NewWithT(t)
		certDir := t.TempDir()
		opts := MetricsServerOptions(":8443", true, certDir, tlsOpts)

		g.Expect(opts.SecureServing).Should(BeTrue())
		g.Expect(opts.FilterProvider).ShouldNot(BeNil())
		g.Expect(opts.CertDir).Should(Equal(certDir))
		g.Expect(opts.TLSOpts).Should(HaveLen(2))

		cfg := &tls.Config{}
		for _, opt := range opts.TLSOpts {
			opt(cfg)
		}
		g.Expect(cfg.GetCertificate).ShouldNot(BeNil())
		_, err := cfg.GetCertificate(&tls.ClientHelloInfo{})
		g.Expect(err).Should(HaveOccurred())

		g.Expect(writeTestServingCerts(certDir)).Should(Succeed())
		cert, err := cfg.GetCertificate(&tls.ClientHelloInfo{})
		g.Expect(err).ShouldNot(HaveOccurred())
		g.Expect(cert.Certificate).ShouldNot(BeEmpty())
	})
}

func writeTestServingCerts(dir string) error {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return fmt.Errorf("generating key: %w", err)
	}
	tmpl := &x509.Certificate{
		SerialNumber: big.NewInt(1),
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(time.Hour),
		DNSNames:     []string{"localhost"},
		KeyUsage:     x509.KeyUsageDigitalSignature,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return fmt.Errorf("creating certificate: %w", err)
	}
	keyBytes, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		return fmt.Errorf("marshalling key: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tls.crt"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600); err != nil {
		return fmt.Errorf("writing tls.crt: %w", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "tls.key"), pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyBytes}), 0o600); err != nil {
		return fmt.Errorf("writing tls.key: %w", err)
	}
	return nil
}
