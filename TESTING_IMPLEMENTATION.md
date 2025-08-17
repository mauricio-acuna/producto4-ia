# 🧪 Portal 4 - Automated Testing Implementation

## 📋 Overview

Implementación de testing automatizado comprehensivo para validar código, documentación y experiencia de usuario siguiendo estándares industriales de OpenAI, Google y Microsoft.

---

## 🏗️ Testing Architecture

### Estructura de Testing
```
tests/
├── unit/                    # Unit tests para funciones individuales
│   ├── test_algorithms.py   # Tests de algoritmos ML
│   ├── test_data_utils.py   # Tests de utilidades de datos
│   └── test_models.py       # Tests de modelos
├── integration/             # Integration tests
│   ├── test_pipelines.py    # Tests de pipelines completos
│   └── test_apis.py         # Tests de APIs
├── e2e/                     # End-to-end tests
│   ├── test_capstones.py    # Tests de proyectos completos
│   └── test_workflows.py    # Tests de workflows de usuario
├── docs/                    # Documentation tests
│   ├── test_links.py        # Validación de enlaces
│   ├── test_code_blocks.py  # Validación de code blocks
│   └── test_notebooks.py    # Validación de notebooks
├── performance/             # Performance tests
│   ├── test_load_times.py   # Tests de velocidad de carga
│   └── test_scalability.py  # Tests de escalabilidad
└── conftest.py             # Configuración compartida
```

---

## ⚙️ Configuration Files

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --tb=short
    --strict-markers
    --disable-warnings
    --cov=src
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=85
    --maxfail=5
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Tests que toman más de 5 segundos
    gpu: Tests que requieren GPU
    network: Tests que requieren conexión a internet
```

### requirements-test.txt
```txt
# Testing Framework
pytest>=7.4.0
pytest-cov>=4.1.0
pytest-xdist>=3.3.1
pytest-timeout>=2.1.0
pytest-mock>=3.11.1

# Code Quality
black>=23.7.0
flake8>=6.0.0
mypy>=1.5.0
isort>=5.12.0
bandit>=1.7.5

# Documentation Testing
linkchecker>=10.2.1
markdown>=3.4.4
beautifulsoup4>=4.12.2

# Performance Testing
pytest-benchmark>=4.0.0
memory-profiler>=0.61.0

# Notebook Testing
nbval>=0.10.0
jupyter>=1.0.0

# Data Testing
pandas>=2.0.0
numpy>=1.24.0
great-expectations>=0.17.0
```

---

## 🧪 Core Testing Implementation

### tests/conftest.py
```python
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
```

### tests/unit/test_algorithms.py
```python
"""Tests unitarios para algoritmos ML implementados desde cero."""
import pytest
import numpy as np
from unittest.mock import patch
import sys
from pathlib import Path

# Agregar src al path para imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from algorithms.regresion_lineal import RegresionLineal
from algorithms.red_neuronal import RedNeuronal
from utils.metricas import calcular_mse, calcular_r2

class TestRegresionLineal:
    """Suite de tests para RegresionLineal."""
    
    def setup_method(self):
        """Setup antes de cada test."""
        self.modelo = RegresionLineal()
        
    def test_inicializacion(self):
        """Test inicialización correcta del modelo."""
        assert self.modelo.theta is None
        assert len(self.modelo.costo_historial) == 0
        assert self.modelo.tasa_aprendizaje == 0.01
        assert self.modelo.max_iteraciones == 1000
        
    def test_agregar_intercepto(self):
        """Test agregado de columna de intercepto."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        X_con_intercepto = self.modelo.agregar_intercepto(X)
        
        assert X_con_intercepto.shape == (3, 3)
        np.testing.assert_array_equal(X_con_intercepto[:, 0], np.ones(3))
        np.testing.assert_array_equal(X_con_intercepto[:, 1:], X)
        
    def test_entrenamiento_datos_lineales_perfectos(self):
        """Test con datos lineales perfectos (sin ruido)."""
        # y = 2x + 1 (perfectamente lineal)
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([3, 5, 7, 9, 11])  # 2x + 1
        
        self.modelo.entrenar(X, y)
        
        # Verificar convergencia a parámetros reales
        np.testing.assert_allclose(self.modelo.theta[0], 1.0, atol=0.1)  # intercepto
        np.testing.assert_allclose(self.modelo.theta[1], 2.0, atol=0.1)  # pendiente
        
        # Verificar que el costo disminuye
        assert self.modelo.costo_historial[0] > self.modelo.costo_historial[-1]
        
    def test_predicciones_precision(self):
        """Test precisión de predicciones."""
        X = np.random.randn(100, 3)
        y = X[:, 0] * 2 + X[:, 1] * -1 + X[:, 2] * 0.5 + np.random.randn(100) * 0.1
        
        self.modelo.entrenar(X, y)
        predicciones = self.modelo.predecir(X)
        
        mse = calcular_mse(y, predicciones)
        r2 = calcular_r2(y, predicciones)
        
        assert mse < 1.0  # MSE razonable
        assert r2 > 0.8   # R² alto para datos sintéticos
        
    def test_dimensiones_incompatibles(self):
        """Test manejo de dimensiones incompatibles."""
        X = np.array([[1, 2], [3, 4]])  # 2x2
        y = np.array([1])  # 1x1
        
        with pytest.raises(ValueError, match="dimensiones incompatibles"):
            self.modelo.entrenar(X, y)
            
    def test_datos_vacios(self):
        """Test manejo de arrays vacíos."""
        X_vacio = np.array([]).reshape(0, 2)
        y_vacio = np.array([])
        
        with pytest.raises(ValueError, match="datos vacíos"):
            self.modelo.entrenar(X_vacio, y_vacio)
            
    @pytest.mark.parametrize("ruido_level", [0.1, 0.5, 1.0, 2.0])
    def test_robustez_ruido(self, ruido_level):
        """Test robustez con diferentes niveles de ruido."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y_clean = X[:, 0] * 3 + X[:, 1] * -2 + 1
        y_noisy = y_clean + np.random.randn(200) * ruido_level
        
        self.modelo.entrenar(X, y_noisy)
        r2 = self.modelo.puntuar(X, y_noisy)
        
        # R² debe degradarse gradualmente con más ruido
        if ruido_level <= 0.5:
            assert r2 > 0.7
        elif ruido_level <= 1.0:
            assert r2 > 0.5
        else:
            assert r2 > 0.2  # Aún debe capturar algo de signal
            
    def test_convergencia_diferentes_learning_rates(self):
        """Test convergencia con diferentes learning rates."""
        X = np.random.randn(50, 2)
        y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(50) * 0.1
        
        learning_rates = [0.001, 0.01, 0.1, 1.0]
        resultados = []
        
        for lr in learning_rates:
            modelo = RegresionLineal(tasa_aprendizaje=lr)
            modelo.entrenar(X, y)
            r2_final = modelo.puntuar(X, y)
            resultados.append(r2_final)
            
        # Debe haber al menos un learning rate que funcione bien
        assert max(resultados) > 0.8
        
    @pytest.mark.slow
    def test_performance_dataset_grande(self):
        """Test performance con dataset grande."""
        n_samples = 10000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        y = np.sum(X[:, :5], axis=1) + np.random.randn(n_samples) * 0.1
        
        import time
        start_time = time.time()
        
        self.modelo.entrenar(X, y)
        
        training_time = time.time() - start_time
        
        # Debe entrenar en tiempo razonable (< 10 segundos)
        assert training_time < 10.0
        assert self.modelo.puntuar(X, y) > 0.7

class TestRedNeuronal:
    """Tests para implementación de red neuronal desde cero."""
    
    def setup_method(self):
        """Setup antes de cada test."""
        self.red = RedNeuronal(capas=[2, 3, 1])
        
    def test_inicializacion_pesos(self):
        """Test inicialización correcta de pesos."""
        assert len(self.red.pesos) == 2  # 2 matrices de pesos
        assert self.red.pesos[0].shape == (2, 3)  # entrada -> oculta
        assert self.red.pesos[1].shape == (3, 1)  # oculta -> salida
        
        # Pesos deben estar en rango razonable
        for peso_matriz in self.red.pesos:
            assert np.all(np.abs(peso_matriz) < 1.0)
            
    def test_propagacion_adelante(self):
        """Test propagación hacia adelante."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        salidas = self.red.propagar_adelante(X)
        
        assert salidas.shape == (4, 1)
        assert np.all(salidas >= 0) and np.all(salidas <= 1)  # sigmoid output
        
    def test_entrenamiento_xor(self):
        """Test entrenamiento en problema XOR."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])  # XOR
        
        costo_inicial = self.red.calcular_costo(X, y)
        self.red.entrenar(X, y, epochs=1000)
        costo_final = self.red.calcular_costo(X, y)
        
        # El costo debe disminuir significativamente
        assert costo_final < costo_inicial * 0.5
        
        # Debe aprender XOR razonablemente bien
        predicciones = self.red.predecir(X)
        precision = np.mean((predicciones > 0.5) == (y > 0.5))
        assert precision > 0.8

class TestMetricas:
    """Tests para funciones de métricas."""
    
    def test_mse_perfecto(self):
        """Test MSE con predicciones perfectas."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        mse = calcular_mse(y_true, y_pred)
        assert mse == 0.0
        
    def test_mse_conocido(self):
        """Test MSE con valor conocido."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([2, 3, 4])  # +1 en cada predicción
        
        mse = calcular_mse(y_true, y_pred)
        assert mse == 1.0  # (1² + 1² + 1²) / 3 = 1
        
    def test_r2_perfecto(self):
        """Test R² con predicciones perfectas."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        
        r2 = calcular_r2(y_true, y_pred)
        assert r2 == 1.0
        
    def test_r2_prediccion_media(self):
        """Test R² cuando predecimos siempre la media."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([3, 3, 3, 3, 3])  # siempre la media
        
        r2 = calcular_r2(y_true, y_pred)
        assert r2 == 0.0
        
    def test_dimensiones_incompatibles_metricas(self):
        """Test manejo de dimensiones incompatibles en métricas."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2])  # diferente longitud
        
        with pytest.raises(ValueError):
            calcular_mse(y_true, y_pred)
            
        with pytest.raises(ValueError):
            calcular_r2(y_true, y_pred)
```

### tests/docs/test_documentation.py
```python
"""Tests para validar la calidad de la documentación."""
import re
import requests
from pathlib import Path
from typing import List, Set
from urllib.parse import urljoin, urlparse
import pytest
import markdown
from bs4 import BeautifulSoup

class TestDocumentationStructure:
    """Tests para estructura de documentación."""
    
    def test_all_modules_have_readme(self):
        """Verifica que todos los módulos tengan README."""
        modulos_path = Path("modulos")
        module_dirs = [d for d in modulos_path.iterdir() 
                      if d.is_dir() and d.name.startswith("modulo-")]
        
        assert len(module_dirs) >= 6, "Debe haber al menos 6 módulos (A-F)"
        
        for module_dir in module_dirs:
            readme_path = module_dir / "README.md"
            assert readme_path.exists(), f"Falta README en {module_dir.name}"
            
            # README no debe estar vacío
            content = readme_path.read_text(encoding="utf-8")
            assert len(content.strip()) > 100, f"README muy corto en {module_dir.name}"
            
    def test_consistent_module_structure(self):
        """Verifica estructura consistente entre módulos."""
        modulos_path = Path("modulos")
        expected_sections = ["Contenido del Módulo", "Actividades", "Recursos"]
        
        for module_dir in modulos_path.iterdir():
            if module_dir.is_dir() and module_dir.name.startswith("modulo-"):
                readme_path = module_dir / "README.md"
                content = readme_path.read_text(encoding="utf-8")
                
                for section in expected_sections:
                    assert section in content, f"Falta sección '{section}' en {module_dir.name}"
                    
    def test_navigation_consistency(self):
        """Verifica que la navegación sea consistente."""
        main_readme = Path("README.md")
        modules_readme = Path("modulos/README.md")
        
        assert main_readme.exists()
        assert modules_readme.exists()
        
        # Verificar links a módulos en README principal
        main_content = main_readme.read_text(encoding="utf-8")
        module_links = re.findall(r'\[.*?\]\((\.\/modulos\/modulo-[a-f]-.*?\/)\)', main_content)
        
        assert len(module_links) >= 6, "Debe haber links a todos los módulos A-F"

class TestLinkValidation:
    """Tests para validación de enlaces."""
    
    def get_all_markdown_files(self) -> List[Path]:
        """Obtiene todos los archivos markdown del proyecto."""
        markdown_files = []
        for pattern in ["**/*.md"]:
            markdown_files.extend(Path(".").glob(pattern))
        return markdown_files
        
    def extract_links(self, content: str) -> List[str]:
        """Extrae todos los enlaces de un contenido markdown."""
        # Enlaces formato [texto](url)
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        # Enlaces formato <url>
        links.extend(re.findall(r'<(https?://[^>]+)>', content))
        return links
        
    def test_internal_links_validity(self):
        """Verifica que todos los enlaces internos sean válidos."""
        broken_links = []
        
        for md_file in self.get_all_markdown_files():
            content = md_file.read_text(encoding="utf-8")
            links = self.extract_links(content)
            
            for link in links:
                if link.startswith(("./", "../", "/")):
                    # Link relativo o absoluto interno
                    if link.startswith("/"):
                        target_path = Path("." + link)
                    else:
                        target_path = (md_file.parent / link).resolve()
                        
                    if not target_path.exists():
                        broken_links.append(f"{md_file}: {link}")
                        
        assert len(broken_links) == 0, f"Enlaces rotos encontrados:\n" + "\n".join(broken_links)
        
    @pytest.mark.network
    def test_external_links_accessibility(self):
        """Verifica que enlaces externos sean accesibles."""
        external_links = set()
        
        for md_file in self.get_all_markdown_files():
            content = md_file.read_text(encoding="utf-8")
            links = self.extract_links(content)
            
            for link in links:
                if link.startswith(("http://", "https://")):
                    external_links.add(link)
                    
        # Limitar a 10 enlaces para no sobrecargar
        test_links = list(external_links)[:10]
        broken_external = []
        
        for link in test_links:
            try:
                response = requests.head(link, timeout=10, allow_redirects=True)
                if response.status_code >= 400:
                    broken_external.append(f"{link}: HTTP {response.status_code}")
            except requests.RequestException as e:
                broken_external.append(f"{link}: {str(e)}")
                
        if broken_external:
            print(f"Enlaces externos con problemas:\n" + "\n".join(broken_external))
            # No fallar el test por enlaces externos, solo reportar

class TestCodeBlocks:
    """Tests para validación de bloques de código."""
    
    def test_code_blocks_have_language(self):
        """Verifica que todos los code blocks tengan lenguaje especificado."""
        issues = []
        
        for md_file in Path(".").glob("**/*.md"):
            content = md_file.read_text(encoding="utf-8")
            
            # Encontrar bloques de código
            code_blocks = re.finditer(r'```(\w*)\n', content)
            
            for i, match in enumerate(code_blocks):
                language = match.group(1)
                if not language.strip():
                    line_number = content[:match.start()].count('\n') + 1
                    issues.append(f"{md_file}:{line_number} - Code block sin lenguaje")
                    
        assert len(issues) == 0, f"Code blocks sin lenguaje:\n" + "\n".join(issues)
        
    def test_python_code_syntax(self):
        """Verifica que el código Python tenga sintaxis válida."""
        syntax_errors = []
        
        for md_file in Path(".").glob("**/*.md"):
            content = md_file.read_text(encoding="utf-8")
            
            # Extraer bloques de código Python
            python_blocks = re.finditer(r'```python\n(.*?)```', content, re.DOTALL)
            
            for i, match in enumerate(python_blocks):
                code = match.group(1)
                
                # Skip code blocks que son solo imports o comentarios
                if len(code.strip()) < 10:
                    continue
                    
                try:
                    compile(code, f"{md_file}:block_{i}", "exec")
                except SyntaxError as e:
                    line_start = content[:match.start()].count('\n') + 1
                    syntax_errors.append(
                        f"{md_file}:{line_start} - Syntax error: {e.msg}"
                    )
                    
        assert len(syntax_errors) == 0, f"Errores de sintaxis Python:\n" + "\n".join(syntax_errors)

class TestContentQuality:
    """Tests para calidad de contenido."""
    
    def test_heading_hierarchy(self):
        """Verifica jerarquía correcta de headings."""
        hierarchy_issues = []
        
        for md_file in Path(".").glob("**/*.md"):
            content = md_file.read_text(encoding="utf-8")
            
            # Extraer headings con sus niveles
            headings = re.findall(r'^(#{1,6})\s+(.+)$', content, re.MULTILINE)
            
            prev_level = 0
            for heading_match in headings:
                current_level = len(heading_match[0])
                
                # El primer heading debe ser H1
                if prev_level == 0 and current_level != 1:
                    hierarchy_issues.append(f"{md_file}: Primer heading debe ser H1")
                    
                # No saltar más de un nivel
                elif current_level > prev_level + 1:
                    hierarchy_issues.append(
                        f"{md_file}: Salto de H{prev_level} a H{current_level}"
                    )
                    
                prev_level = current_level
                
        assert len(hierarchy_issues) == 0, f"Problemas de jerarquía:\n" + "\n".join(hierarchy_issues)
        
    def test_minimum_content_length(self):
        """Verifica que los archivos tengan contenido mínimo."""
        short_files = []
        min_words = 100
        
        for md_file in Path(".").glob("**/*.md"):
            if md_file.name.startswith("test_"):
                continue
                
            content = md_file.read_text(encoding="utf-8")
            
            # Contar palabras excluyendo código y metadatos
            text_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            text_content = re.sub(r'`[^`]+`', '', text_content)
            
            words = len(text_content.split())
            
            if words < min_words:
                short_files.append(f"{md_file}: {words} palabras (mínimo {min_words})")
                
        assert len(short_files) == 0, f"Archivos muy cortos:\n" + "\n".join(short_files)
        
    def test_consistent_spanish_language(self):
        """Verifica consistencia en el uso del español."""
        english_patterns = [
            r'\bthe\b', r'\band\b', r'\bor\b', r'\bbut\b', r'\bfor\b',
            r'\bwith\b', r'\bfrom\b', r'\bthis\b', r'\bthat\b', r'\bwhen\b'
        ]
        
        issues = []
        
        for md_file in Path(".").glob("**/*.md"):
            content = md_file.read_text(encoding="utf-8")
            
            # Excluir bloques de código
            text_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
            text_content = re.sub(r'`[^`]+`', '', text_content)
            
            for pattern in english_patterns:
                matches = re.finditer(pattern, text_content, re.IGNORECASE)
                for match in matches:
                    line_num = text_content[:match.start()].count('\n') + 1
                    issues.append(f"{md_file}:{line_num} - Posible texto en inglés: '{match.group()}'")
                    
        # Permitir algunas ocurrencias pero reportar si hay muchas
        if len(issues) > 20:
            print(f"Advertencia: {len(issues)} posibles textos en inglés detectados")
            print("Primeros 10:")
            for issue in issues[:10]:
                print(f"  {issue}")
```

### tests/integration/test_pipelines.py
```python
"""Tests de integración para pipelines completos."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from pipelines.training_pipeline import TrainingPipeline
from pipelines.data_pipeline import DataPipeline
from pipelines.evaluation_pipeline import EvaluationPipeline

class TestTrainingPipeline:
    """Tests para pipeline de entrenamiento completo."""
    
    def setup_method(self):
        """Setup para cada test."""
        self.pipeline = TrainingPipeline()
        
    def test_pipeline_completo_regresion(self, sample_dataset, temp_workspace):
        """Test pipeline completo para problema de regresión."""
        # Guardar dataset en workspace temporal
        data_path = temp_workspace / "dataset.csv"
        sample_dataset.to_csv(data_path, index=False)
        
        # Configurar pipeline
        config = {
            "data_path": str(data_path),
            "target_column": "target",
            "model_type": "linear_regression",
            "test_size": 0.2,
            "random_state": 42
        }
        
        # Ejecutar pipeline
        results = self.pipeline.run(config)
        
        # Verificar resultados
        assert "model" in results
        assert "metrics" in results
        assert "predictions" in results
        
        # Verificar métricas
        metrics = results["metrics"]
        assert "train_mse" in metrics
        assert "test_mse" in metrics
        assert "train_r2" in metrics
        assert "test_r2" in metrics
        
        # Verificar calidad mínima
        assert metrics["test_r2"] > 0.5
        assert metrics["test_mse"] < 2.0
        
    def test_pipeline_manejo_errores(self, temp_workspace):
        """Test manejo de errores en pipeline."""
        # Dataset inválido
        config = {
            "data_path": "archivo_inexistente.csv",
            "target_column": "target",
            "model_type": "linear_regression"
        }
        
        with pytest.raises(FileNotFoundError):
            self.pipeline.run(config)
            
    def test_pipeline_diferentes_modelos(self, sample_dataset, temp_workspace):
        """Test pipeline con diferentes tipos de modelo."""
        data_path = temp_workspace / "dataset.csv"
        sample_dataset.to_csv(data_path, index=False)
        
        modelos = ["linear_regression", "neural_network", "random_forest"]
        resultados = {}
        
        for modelo in modelos:
            config = {
                "data_path": str(data_path),
                "target_column": "target",
                "model_type": modelo,
                "test_size": 0.2
            }
            
            try:
                results = self.pipeline.run(config)
                resultados[modelo] = results["metrics"]["test_r2"]
            except Exception as e:
                pytest.fail(f"Error con modelo {modelo}: {e}")
                
        # Al menos un modelo debe funcionar bien
        assert max(resultados.values()) > 0.6

class TestDataPipeline:
    """Tests para pipeline de procesamiento de datos."""
    
    def test_cleaning_pipeline(self, temp_workspace):
        """Test pipeline de limpieza de datos."""
        # Crear dataset sucio
        dirty_data = pd.DataFrame({
            'feature_1': [1, 2, np.nan, 4, 5],
            'feature_2': [1, 2, 3, 4, np.inf],
            'feature_3': ['a', 'b', 'c', 'd', 'e'],
            'target': [1, 2, 3, 4, 5]
        })
        
        data_path = temp_workspace / "dirty_data.csv"
        dirty_data.to_csv(data_path, index=False)
        
        pipeline = DataPipeline()
        
        config = {
            "input_path": str(data_path),
            "operations": [
                "handle_missing_values",
                "handle_infinite_values",
                "encode_categorical",
                "normalize_features"
            ]
        }
        
        cleaned_data = pipeline.run(config)
        
        # Verificar limpieza
        assert not cleaned_data.isnull().any().any()
        assert not np.isinf(cleaned_data.select_dtypes(include=[np.number]).values).any()
        assert cleaned_data.shape[0] > 0

class TestEvaluationPipeline:
    """Tests para pipeline de evaluación."""
    
    def test_evaluation_completa(self, sample_dataset, ml_config):
        """Test evaluación completa de modelo."""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        
        # Preparar datos
        X = sample_dataset.drop('target', axis=1)
        y = sample_dataset['target']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=ml_config['test_size'], 
            random_state=ml_config['random_state']
        )
        
        # Entrenar modelo
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluar
        pipeline = EvaluationPipeline()
        
        config = {
            "model": model,
            "X_test": X_test,
            "y_test": y_test,
            "X_train": X_train,
            "y_train": y_train,
            "metrics": ["mse", "r2", "mae"],
            "cross_validation": True,
            "cv_folds": 5
        }
        
        results = pipeline.run(config)
        
        # Verificar resultados
        assert "test_metrics" in results
        assert "train_metrics" in results
        assert "cv_scores" in results
        
        # Verificar métricas específicas
        test_metrics = results["test_metrics"]
        assert "mse" in test_metrics
        assert "r2" in test_metrics
        assert "mae" in test_metrics
        
        # Cross-validation debe tener scores razonables
        cv_scores = results["cv_scores"]
        assert len(cv_scores) == 5
        assert all(score > 0.3 for score in cv_scores)  # R² mínimo razonable
```

---

## 🚀 GitHub Actions CI/CD

### .github/workflows/tests.yml
```yaml
name: Portal 4 - Automated Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11]
        
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install -r requirements.txt
        
    - name: Lint with flake8
      run: |
        flake8 src tests --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src tests --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
        
    - name: Format check with black
      run: |
        black --check src tests
        
    - name: Type checking with mypy
      run: |
        mypy src --ignore-missing-imports
        
    - name: Security check with bandit
      run: |
        bandit -r src
        
    - name: Test with pytest
      run: |
        pytest tests/ -v --cov=src --cov-report=xml --cov-report=term-missing
        
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        
  docs-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        
    - name: Test documentation
      run: |
        pytest tests/docs/ -v -m "not network"
        
    - name: Check external links
      run: |
        pytest tests/docs/test_documentation.py::TestLinkValidation::test_external_links_accessibility -v
      continue-on-error: true
      
  notebook-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        pip install jupyter nbval
        
    - name: Test notebooks
      run: |
        find . -name "*.ipynb" -not -path "./.*" | xargs pytest --nbval-lax
```

### .github/workflows/quality-gates.yml
```yaml
name: Quality Gates

on:
  pull_request:
    branches: [ main ]

jobs:
  quality-check:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: 3.11
        
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements-test.txt
        
    - name: Check test coverage threshold
      run: |
        pytest tests/ --cov=src --cov-fail-under=85
        
    - name: Check code complexity
      run: |
        radon cc src --min B  # Complexity threshold
        
    - name: Check documentation coverage
      run: |
        interrogate src --ignore-init-method --ignore-module --fail-under=80
        
    - name: Performance regression test
      run: |
        pytest tests/performance/ --benchmark-only
        
    - name: Security vulnerability scan
      run: |
        safety check
        bandit -r src -f json -o bandit-report.json
        
    - name: Generate quality report
      run: |
        python scripts/generate_quality_report.py
        
    - name: Comment PR with results
      uses: actions/github-script@v6
      with:
        script: |
          const fs = require('fs');
          const report = fs.readFileSync('quality-report.md', 'utf8');
          github.rest.issues.createComment({
            issue_number: context.issue.number,
            owner: context.repo.owner,
            repo: context.repo.repo,
            body: report
          });
```

---

## 📊 Quality Dashboard

### scripts/generate_quality_report.py
```python
"""Genera reporte de calidad comprehensivo."""
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_command(cmd: str) -> str:
    """Ejecuta comando y retorna output."""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def get_test_coverage():
    """Obtiene cobertura de tests."""
    coverage_data = run_command("coverage json")
    if coverage_data:
        data = json.loads(coverage_data)
        return data["totals"]["percent_covered"]
    return 0

def get_code_quality_metrics():
    """Obtiene métricas de calidad de código."""
    # Complexity
    complexity = run_command("radon cc src --json")
    
    # Maintainability
    maintainability = run_command("radon mi src --json")
    
    # Lines of code
    loc = run_command("cloc src --json")
    
    return {
        "complexity": complexity,
        "maintainability": maintainability,
        "lines_of_code": loc
    }

def generate_report():
    """Genera reporte completo."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    coverage = get_test_coverage()
    code_metrics = get_code_quality_metrics()
    
    report = f"""
# 📊 Portal 4 - Quality Report

**Generated:** {timestamp}

## 🧪 Test Coverage
- **Overall Coverage:** {coverage:.1f}%
- **Target:** 85%+
- **Status:** {'✅ PASS' if coverage >= 85 else '❌ FAIL'}

## 📏 Code Quality Metrics

### Test Results Summary
```bash
{run_command("pytest tests/ --tb=no -q")}
```

### Security Scan
```bash
{run_command("bandit -r src -f txt")}
```

### Documentation Coverage
```bash
{run_command("interrogate src")}
```

## 📈 Trends
- Coverage trend: [Implementar tracking histórico]
- Performance trend: [Implementar benchmarking]
- Issue resolution rate: [Implementar tracking]

## 🎯 Action Items
- [ ] Increase coverage to 85%+ if below threshold
- [ ] Address high complexity functions
- [ ] Fix security vulnerabilities if any
- [ ] Update documentation for undocumented functions

---
*This report is automatically generated by Portal 4 Quality System*
"""
    
    Path("quality-report.md").write_text(report)
    print("✅ Quality report generated: quality-report.md")

if __name__ == "__main__":
    generate_report()
```

---

## 🎯 Implementation Checklist

### Phase 1: Core Testing (Week 1)
- [ ] Setup pytest configuration and dependencies
- [ ] Implement unit tests for all algorithms
- [ ] Create integration tests for pipelines
- [ ] Setup GitHub Actions CI/CD
- [ ] Configure coverage reporting

### Phase 2: Documentation Testing (Week 2)
- [ ] Implement link validation tests
- [ ] Create code block syntax validation
- [ ] Setup content quality checks
- [ ] Implement notebook testing
- [ ] Configure automated reporting

### Phase 3: Quality Gates (Week 3)
- [ ] Setup coverage thresholds (85%+)
- [ ] Implement complexity monitoring
- [ ] Configure security scanning
- [ ] Setup performance regression testing
- [ ] Create quality dashboard

### Phase 4: Advanced Testing (Week 4)
- [ ] Implement property-based testing
- [ ] Setup load testing for documentation site
- [ ] Create user experience testing
- [ ] Implement accessibility testing
- [ ] Configure monitoring and alerting

**Objetivo:** Portal 4 con 95%+ de confiabilidad técnica, matching OpenAI Cookbook standards y superando expectativas industriales.
