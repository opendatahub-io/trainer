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
import os
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread


class MetricsHandler(BaseHTTPRequestHandler):
    progress_data = {
        "progressPercentage": 0,
        "estimatedRemainingSeconds": 50,
        "currentStep": 0,
        "totalSteps": 50,
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


metrics_port = int(os.environ.get("RHAI_E2E_METRICS_PORT", "28080"))
start_metrics_server(metrics_port)
time.sleep(1)

print("Starting training...")
total_steps = 50
for step in range(total_steps + 1):
    time.sleep(0.2)
    progress = int((step / total_steps) * 100)
    remaining = int((total_steps - step) * 0.2)

    MetricsHandler.progress_data = {
        "progressPercentage": progress,
        "estimatedRemainingSeconds": remaining,
        "currentStep": step,
        "totalSteps": total_steps,
        "currentEpoch": 1,
        "totalEpochs": 1,
        "trainMetrics": {"loss": 1.0 - (progress / 100.0)},
        "evalMetrics": {"accuracy": 0.5 + (progress / 200.0)},
    }
    if step % 10 == 0:
        print(f"Step {step}/{total_steps}, Progress {progress}%")

print("Training completed!")
time.sleep(10)
