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

package support

import (
	"context"
	"fmt"
	"io"
	"sort"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"

	componentsv1alpha1 "github.com/hrathina/odh-trainer-operator/api/v1alpha1"
)

const trainerCRName = "default-trainer"

type Client struct {
	kubernetes.Interface
	CRClient client.Client
}

func NewClient() (*Client, error) {
	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(
		clientcmd.NewDefaultClientConfigLoadingRules(),
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, err
	}
	clientset, err := kubernetes.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}

	scheme := runtime.NewScheme()
	if err := componentsv1alpha1.AddToScheme(scheme); err != nil {
		return nil, err
	}

	crClient, err := client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		return nil, err
	}

	return &Client{Interface: clientset, CRClient: crClient}, nil
}

func (c *Client) GetPodLogs(ctx context.Context, name, ns string) (string, error) {
	req := c.CoreV1().Pods(ns).GetLogs(name, &corev1.PodLogOptions{})
	stream, err := req.Stream(ctx)
	if err != nil {
		return "", err
	}
	defer func() { _ = stream.Close() }()
	buf, err := io.ReadAll(stream)
	if err != nil {
		return "", err
	}
	return string(buf), nil
}

func (c *Client) RegisterDebugCleanup(t *testing.T, ctx context.Context, ns string, extraPods ...string) {
	t.Helper()
	t.Cleanup(func() {
		if !t.Failed() {
			return
		}

		controllerLogs, err := c.GetControllerLogs(ctx, ns)
		if err == nil {
			t.Logf("Controller logs:\n %s", controllerLogs)
		} else {
			t.Logf("Failed to get Controller logs: %s", err)
		}

		events, err := c.CoreV1().Events(ns).List(ctx, metav1.ListOptions{})
		if err == nil {
			sort.Slice(events.Items, func(i, j int) bool {
				return events.Items[i].LastTimestamp.Time.Before(events.Items[j].LastTimestamp.Time)
			})
			var lines []string
			for _, e := range events.Items {
				lines = append(lines, fmt.Sprintf("%s\t%s\t%s\t%s\t%s",
					e.LastTimestamp.Format(time.RFC3339),
					e.Type, e.Reason, e.InvolvedObject.Name, e.Message,
				))
			}
			t.Logf("Kubernetes events:\n%s", strings.Join(lines, "\n"))
		} else {
			t.Logf("Failed to get Kubernetes events: %s", err)
		}

		for _, pod := range extraPods {
			output, err := c.GetPodLogs(ctx, pod, ns)
			if err == nil {
				t.Logf("%s logs:\n %s", pod, output)
			} else {
				t.Logf("Failed to get %s logs: %s", pod, err)
			}
		}
	})
}

func (c *Client) GetControllerLogs(ctx context.Context, ns string) (string, error) {
	pods, err := c.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{
		LabelSelector: "control-plane=controller-manager",
	})
	if err != nil {
		return "", err
	}
	if len(pods.Items) == 0 {
		return "", fmt.Errorf("no controller-manager pods found")
	}
	return c.GetPodLogs(ctx, pods.Items[0].Name, ns)
}

func (c *Client) CreateTrainer(ctx context.Context, managementState common.ManagementState, appNamespace string) error {
	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: trainerCRName,
		},
		Spec: componentsv1alpha1.TrainerSpec{
			ManagementState: managementState,
			AppNamespace:    appNamespace,
		},
	}
	return c.CRClient.Create(ctx, trainer)
}

func (c *Client) GetTrainer(ctx context.Context) (*componentsv1alpha1.Trainer, error) {
	trainer := &componentsv1alpha1.Trainer{}
	err := c.CRClient.Get(ctx, client.ObjectKey{Name: trainerCRName}, trainer)
	return trainer, err
}

func (c *Client) DeleteTrainer(ctx context.Context) error {
	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: trainerCRName,
		},
	}
	return client.IgnoreNotFound(c.CRClient.Delete(ctx, trainer))
}
