# Adding a New Model or Method to the Python SDK

## When to use

Use this skill when adding or updating Python SDK models (Pydantic types for Kubernetes resources) or adding new initializer providers for dataset/model downloading.

The Python SDK has two distinct parts:
1. **Auto-generated API models** (`api/python_api/`) — Pydantic models generated from the OpenAPI spec
2. **Initializers** (`pkg/initializers/`) — hand-written dataset and model download providers

## Part A: Updating auto-generated API models

### When to do this

After modifying Go API types in `pkg/apis/trainer/v1alpha1/`, the Python models must be regenerated to stay in sync.

### Steps

1. **Update the Go API types** in `pkg/apis/trainer/v1alpha1/` (add/modify fields, types, markers)

2. **Regenerate the OpenAPI spec and Python models**:
   ```bash
   make generate
   ```
   This runs the full generation pipeline:
   - `controller-gen` generates CRD manifests and OpenAPI schema
   - `hack/update-codegen.sh` regenerates Go clients
   - `hack/python-api/gen-api.sh` regenerates Python SDK models

3. **What gets generated**:
   - `api/openapi-spec/swagger.json` — updated OpenAPI spec
   - `api/python_api/kubeflow_trainer_api/models/` — 372+ Pydantic V2 model files
   - `api/python_api/kubeflow_trainer_api/__init__.py` — model index with version

4. **Do NOT manually edit** files in `api/python_api/kubeflow_trainer_api/models/` — they are overwritten on every generation run

### Generation config

- Config file: `hack/python-api/swagger_config.json`
- Generator: OpenAPI Generator v7.13.0 (runs in a container via Docker/Podman)
- Target: Pydantic V2 models with Python 3.9+ compatibility

### Model structure

Each generated model follows this pattern:
```python
class TrainerKubeflowOrgV1alpha1TrainJobSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True, protected_namespaces=())

    runtime_ref: RuntimeRef = Field(alias="runtimeRef")
    trainer: Optional[Trainer] = None
    # ...

    def to_dict(self) -> Dict[str, Any]: ...
    def from_dict(cls, obj: Dict[str, Any]) -> Self: ...
```

### Package metadata

- Package name: `kubeflow_trainer_api`
- Version: defined in `api/python_api/kubeflow_trainer_api/__init__.py`
- Build system: hatchling (`pyproject.toml`)
- Dependencies: `pydantic>=2.10.0`

## Part B: Adding a new initializer provider

### When to do this

When adding support for a new storage backend (e.g., GCS, Azure Blob) or a new model/dataset source for the init containers.

### Steps

#### 1. Define the configuration dataclass

Add a new dataclass in `pkg/initializers/types/types.py`:

```python
@dataclass
class GCSDatasetInitializer:
    storage_uri: str = ""
    ignore_patterns: list = field(default_factory=list)
    project_id: str = ""
    credentials_json: str = ""
```

Field names are converted to UPPERCASE environment variables automatically by `get_config_from_env()`.

#### 2. Implement the provider

Create `pkg/initializers/dataset/<provider>.py` (or `model/<provider>.py`):

```python
from pkg.initializers.utils.utils import DatasetProvider, DATASET_PATH

class GCSDatasetProvider(DatasetProvider):
    def load_config(self):
        self._config = get_config_from_env(GCSDatasetInitializer)

    def download_dataset(self):
        # Implement download logic
        # Download to DATASET_PATH (/workspace/dataset)
        pass
```

Follow the existing pattern in:
- `pkg/initializers/dataset/huggingface.py` — HuggingFace provider
- `pkg/initializers/dataset/s3.py` — S3 provider using OpenDAL
- `pkg/initializers/model/huggingface.py` — model variant

#### 3. Register the URI scheme

Update `pkg/initializers/dataset/__main__.py` (or `model/__main__.py`) to handle the new scheme:

```python
match urlparse(storage_uri).scheme:
    case "hf":
        provider = HuggingFaceDatasetProvider()
    case "s3":
        provider = S3DatasetProvider()
    case "gcs":                              # new scheme
        provider = GCSDatasetProvider()
    case _:
        raise ValueError(f"Invalid scheme: ...")
```

The URI scheme constants are defined in `pkg/initializers/utils/utils.py`.

#### 4. Add requirements

If the new provider needs additional Python packages, add them to the appropriate requirements file:
- Dataset: `cmd/initializers/dataset/requirements.txt`
- Model: `cmd/initializers/model/requirements.txt`

#### 5. Write tests

Add unit tests in `pkg/initializers/dataset/<provider>_test.py`:

```python
import pytest
from pkg.initializers.dataset.<provider> import GCSDatasetProvider

class TestGCSDatasetProvider:
    @pytest.mark.parametrize("test_name,config,expected", [...])
    def test_load_config(self, test_name, config, expected):
        # Test config loading from env vars
        pass

    def test_download_dataset(self):
        # Test download logic with mocked client
        pass
```

Follow the testing patterns in:
- `pkg/initializers/dataset/huggingface_test.py`
- `pkg/initializers/dataset/s3_test.py`
- `pkg/initializers/utils/utils_test.py`

Run tests:
```bash
make test-python              # Unit tests
make test-python-integration  # Integration tests
```

#### 6. Update Go constants (if needed)

If the new provider introduces new URI schemes or container names, update `pkg/constants/constants.go`.

## Key patterns

- **Provider pattern**: All providers inherit from `DatasetProvider` or `ModelProvider` abstract base classes
- **Config from env**: `get_config_from_env()` reflectively maps dataclass fields to env vars
- **URI scheme dispatch**: Python 3.10+ `match/case` on `urlparse(storage_uri).scheme`
- **Workspace paths**: Downloads go to `/workspace/dataset` or `/workspace/model`
- **No manual model edits**: The `api/python_api/kubeflow_trainer_api/models/` directory is fully generated