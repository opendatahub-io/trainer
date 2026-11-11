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
	"crypto/tls"
	"testing"

	. "github.com/onsi/gomega"
)

func TestWebhookServerOptionsAppliesCurvePreferences(t *testing.T) {
	g := NewWithT(t)
	options := WebhookServerOptions([]func(*tls.Config){func(config *tls.Config) {
		config.CurvePreferences = []tls.CurveID{tls.X25519}
	}})
	g.Expect(options.TLSOpts).To(HaveLen(1))

	config := &tls.Config{}
	for _, configure := range options.TLSOpts {
		configure(config)
	}
	g.Expect(config.CurvePreferences).To(Equal([]tls.CurveID{tls.X25519}))
}
