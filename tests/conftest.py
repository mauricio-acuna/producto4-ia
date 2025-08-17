"""Configuración compartida para todos los tests."""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator, Dict, Any

# Configurar environment variables para testing
os.environ["TESTING"] = "true"
os.environ["LOG_LEVEL"] = "ERROR"

@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Directorio con datos de prueba."""
    return Path(__file__).parent / "test_data"

@pytest.fixture(scope="session")
def sample_dataset() -> pd.DataFrame:
    """Dataset sintético para testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Relación lineal con ruido
    y = X[:, 0] * 2 + X[:, 1] * -1.5 + np.random.randn(n_samples) * 0.1
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target'] = y
    return df

@pytest.fixture
def temp_workspace() -> Generator[Path, None, None]:
    """Workspace temporal para tests que crean archivos."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def ml_config() -> Dict[str, Any]:
    """Configuración estándar para modelos ML."""
    return {
        "test_size": 0.2,
        "random_state": 42,
        "max_iterations": 100,
        "tolerance": 1e-6,
        "learning_rate": 0.01
    }

class MockAPIResponse:
    """Mock para respuestas de API."""
    def __init__(self, json_data: Dict, status_code: int = 200):
        self.json_data = json_data
        self.status_code = status_code
        
    def json(self):
        return self.json_data
        
    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

@pytest.fixture
def mock_openai_response():
    """Mock response para OpenAI API."""
    return MockAPIResponse({
        "choices": [{
            "message": {
                "content": "This is a test response from OpenAI API."
            }
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 15,
            "total_tokens": 25
        }
    })
