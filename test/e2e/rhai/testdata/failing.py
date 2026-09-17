# Copyright 2026 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread


class MetricsHandler(BaseHTTPRequestHandler):
    progress_data = {
        "progressPercentage": 0,
        "estimatedRemainingSeconds": 30,
        "currentStep": 0,
        "totalSteps": 30,
        "currentEpoch": 0,
        "totalEpochs": 1,
        "trainMetrics": {"loss": 1.0},
        "evalMetrics": {},
    }

    def do_GET(self):
        if self.path == "/metrics":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(self.progress_data).encode())
        else:
            self.send_error(404)

    def log_message(self, *args):
        pass


def start_metrics_server(port=28080):
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    Thread(target=server.serve_forever, daemon=True).start()
    print(f"Metrics server started on port {port}")


start_metrics_server(28080)
time.sleep(1)

print("Starting training that will fail...")
total_steps = 30
fail_at_step = 15

for step in range(fail_at_step):
    time.sleep(0.5)
    progress = int((step / total_steps) * 100)
    remaining = int((total_steps - step) * 0.5)

    MetricsHandler.progress_data = {
        "progressPercentage": progress,
        "estimatedRemainingSeconds": remaining,
        "currentStep": step,
        "totalSteps": total_steps,
        "currentEpoch": 1,
        "totalEpochs": 1,
        "trainMetrics": {"loss": 1.0 - (progress / 100.0)},
        "evalMetrics": {},
    }
    if step % 5 == 0:
        print(f"Step {step}/{total_steps}, Progress {progress}%")

print(f"ERROR: Training failed at step {fail_at_step}/{total_steps}")
time.sleep(0.5)
sys.exit(1)
