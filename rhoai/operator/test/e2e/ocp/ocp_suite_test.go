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

package ocp

import (
	"context"
	"fmt"
	"log"
	"os"
	"testing"
	"time"

	"github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/opendatahub-io/trainer-operator/test/support"
)

var (
	k8sClient *support.Client
	ctx       = context.Background()
)

const namespace = "trainer-operator-system"

func TestMain(m *testing.M) {
	fmt.Fprintln(os.Stderr, "Starting OCP e2e test suite")

	gomega.SetDefaultEventuallyTimeout(2 * time.Minute)
	gomega.SetDefaultEventuallyPollingInterval(time.Second)

	var err error
	k8sClient, err = support.NewClient()
	if err != nil {
		log.Fatalf("Failed to create Kubernetes client: %v", err)
	}

	pods, err := k8sClient.CoreV1().Pods(namespace).List(
		ctx, metav1.ListOptions{
			LabelSelector: "control-plane=controller-manager",
		})
	if err != nil {
		log.Fatalf("Failed to list controller pods: %v", err)
	}
	if len(pods.Items) == 0 {
		log.Fatalf("No controller-manager pod found in %s — "+
			"deploy the operator before running OCP e2e tests", namespace)
	}

	os.Exit(m.Run())
}
