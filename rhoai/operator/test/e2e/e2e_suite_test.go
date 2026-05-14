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

package e2e

import (
	"fmt"
	"log"
	"os"
	"os/exec"
	"testing"
	"time"

	"github.com/onsi/gomega"

	"github.com/hrathina/odh-trainer-operator/test/utils"
)

var (
	// Optional Environment Variables:
	// - CERT_MANAGER_INSTALL_SKIP=true: Skips CertManager installation during test setup.
	// These variables are useful if CertManager is already installed, avoiding
	// re-installation and conflicts.
	skipCertManagerInstall = os.Getenv("CERT_MANAGER_INSTALL_SKIP") == "true"
	// isCertManagerAlreadyInstalled will be set true when CertManager CRDs be found on the cluster
	isCertManagerAlreadyInstalled = false

	// projectImage is the name of the image which will be build and loaded
	// with the code source changes to be tested.
	projectImage = "example.com/odh-trainer-operator:v0.0.1"
)

func TestMain(m *testing.M) {
	fmt.Fprintln(os.Stderr, "Starting odh-trainer-operator integration test suite")

	gomega.SetDefaultEventuallyTimeout(2 * time.Minute)
	gomega.SetDefaultEventuallyPollingInterval(time.Second)

	cmd := exec.Command("make", "docker-build", fmt.Sprintf("IMG=%s", projectImage))
	if _, err := utils.Run(cmd); err != nil {
		log.Fatalf("Failed to build the manager(Operator) image: %v", err)
	}

	if err := utils.LoadImageToKindClusterWithName(projectImage); err != nil {
		log.Fatalf("Failed to load the manager(Operator) image into Kind: %v", err)
	}

	if !skipCertManagerInstall {
		isCertManagerAlreadyInstalled = utils.IsCertManagerCRDsInstalled()
		if !isCertManagerAlreadyInstalled {
			fmt.Fprintln(os.Stderr, "Installing CertManager...")
			if err := utils.InstallCertManager(); err != nil {
				log.Fatalf("Failed to install CertManager: %v", err)
			}
		} else {
			fmt.Fprintln(os.Stderr, "WARNING: CertManager is already installed. Skipping installation...")
		}
	}

	code := m.Run()

	if !skipCertManagerInstall && !isCertManagerAlreadyInstalled {
		fmt.Fprintln(os.Stderr, "Uninstalling CertManager...")
		utils.UninstallCertManager()
	}

	os.Exit(code)
}
