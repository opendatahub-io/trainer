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
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/opendatahub-io/odh-platform-utilities/api/common"

	componentsv1alpha1 "github.com/opendatahub-io/trainer-operator/api/v1alpha1"
)

const (
	trainerCRName      = "default-trainer"
	trainerKubeflowAPI = "trainer.kubeflow.org"
	trainerAPIVersion  = "v1alpha1"
	fieldName          = "name"
)

type Client struct {
	kubernetes.Interface
	CRClient            client.Client
	DynamicClient       dynamic.Interface
	ApiextensionsClient apiextensionsclientset.Interface
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

	apiextensionsClient, err := apiextensionsclientset.NewForConfig(cfg)
	if err != nil {
		return nil, err
	}

	dynamicClient, err := dynamic.NewForConfig(cfg)
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

	return &Client{
		Interface:           clientset,
		CRClient:            crClient,
		DynamicClient:       dynamicClient,
		ApiextensionsClient: apiextensionsClient,
	}, nil
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

var clusterTrainingRuntimeGVR = schema.GroupVersionResource{
	Group:    trainerKubeflowAPI,
	Version:  trainerAPIVersion,
	Resource: "clustertrainingruntimes",
}

var trainJobGVR = schema.GroupVersionResource{
	Group:    trainerKubeflowAPI,
	Version:  trainerAPIVersion,
	Resource: "trainjobs",
}

var jobSetGVR = schema.GroupVersionResource{
	Group:    "jobset.x-k8s.io",
	Version:  "v1alpha2",
	Resource: "jobsets",
}

func (c *Client) ListClusterTrainingRuntimes(ctx context.Context, labelSelector string) ([]string, error) {
	list, err := c.DynamicClient.Resource(clusterTrainingRuntimeGVR).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, err
	}

	var names []string
	for _, item := range list.Items {
		names = append(names, item.GetName())
	}
	return names, nil
}

func (c *Client) UpdateTrainerManagementState(ctx context.Context, state common.ManagementState) error {
	trainer, err := c.GetTrainer(ctx)
	if err != nil {
		return err
	}
	trainer.Spec.ManagementState = state
	return c.CRClient.Update(ctx, trainer)
}

func (c *Client) ListDeployments(ctx context.Context, namespace, labelSelector string) ([]string, error) {
	deployments, err := c.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, err
	}

	var names []string
	for _, d := range deployments.Items {
		names = append(names, d.Name)
	}
	return names, nil
}

func (c *Client) DeleteTrainer(ctx context.Context) error {
	trainer := &componentsv1alpha1.Trainer{
		ObjectMeta: metav1.ObjectMeta{
			Name: trainerCRName,
		},
	}
	return client.IgnoreNotFound(c.CRClient.Delete(ctx, trainer))
}

func (c *Client) GetClusterTrainingRuntime(ctx context.Context, name string) (*unstructured.Unstructured, error) {
	return c.DynamicClient.Resource(clusterTrainingRuntimeGVR).Get(ctx, name, metav1.GetOptions{})
}

func (c *Client) CreateTrainJob(ctx context.Context, name, namespace, runtimeName string) error {
	trainJob := &unstructured.Unstructured{}
	trainJob.SetUnstructuredContent(map[string]any{
		"apiVersion": trainerKubeflowAPI + "/" + trainerAPIVersion,
		"kind":       "TrainJob",
		"metadata": map[string]any{
			fieldName:   name,
			"namespace": namespace,
		},
		"spec": map[string]any{
			"runtimeRef": map[string]any{
				fieldName: runtimeName,
			},
		},
	})
	_, err := c.DynamicClient.Resource(trainJobGVR).Namespace(namespace).Create(ctx, trainJob, metav1.CreateOptions{})
	return err
}

func (c *Client) CreateTrainJobWithCommand(
	ctx context.Context,
	name, namespace, runtimeName string,
	command []string,
) error {
	cmd := make([]any, len(command))
	for i, c := range command {
		cmd[i] = c
	}
	trainJob := &unstructured.Unstructured{}
	trainJob.SetUnstructuredContent(map[string]any{
		"apiVersion": trainerKubeflowAPI + "/" + trainerAPIVersion,
		"kind":       "TrainJob",
		"metadata": map[string]any{
			fieldName:   name,
			"namespace": namespace,
		},
		"spec": map[string]any{
			"runtimeRef": map[string]any{
				fieldName: runtimeName,
			},
			"trainer": map[string]any{
				"command": cmd,
			},
		},
	})
	_, err := c.DynamicClient.Resource(trainJobGVR).Namespace(namespace).
		Create(ctx, trainJob, metav1.CreateOptions{})
	return err
}

func (c *Client) GetTrainJob(ctx context.Context, name, namespace string) (*unstructured.Unstructured, error) {
	return c.DynamicClient.Resource(trainJobGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
}

func (c *Client) ListJobSets(ctx context.Context, namespace string) ([]string, error) {
	list, err := c.DynamicClient.Resource(jobSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}

	var names []string
	for _, item := range list.Items {
		names = append(names, item.GetName())
	}
	return names, nil
}

func (c *Client) DeleteTrainJob(ctx context.Context, name, namespace string) error {
	err := c.DynamicClient.Resource(trainJobGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if errors.IsNotFound(err) {
		return nil
	}
	return err
}
