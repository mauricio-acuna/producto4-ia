# 📐 Portal 4 Style Guide & Standards

## 🎯 Objetivo

Establecer estándares de calidad industrial comparable con Google, OpenAI, Microsoft y Apple para que Portal 4 se convierta en referencia de la industria.

---

## 📝 Documentation Standards

### Estructura de Archivos
```
modulos/
├── README.md                 # Overview con navigation clara
├── modulo-{letra}-{nombre}/
│   ├── README.md            # Contenido principal del módulo
│   ├── actividades/         # Ejercicios hands-on
│   ├── recursos/            # Materials de apoyo
│   ├── ejemplos/            # Code samples validados
│   └── evaluacion/          # Rubrics y assessments
```

### Naming Conventions
```yaml
Files: kebab-case (modulo-a-fundamentos.md)
Directories: kebab-case (actividades/, recursos/)
Variables: snake_case (learning_objectives)
Classes: PascalCase (RegresionLineal)
Functions: snake_case (entrenar_modelo)
Constants: UPPER_SNAKE_CASE (API_BASE_URL)
```

### Markdown Standards
```markdown
# H1: Solo uno por archivo (título principal)
## H2: Secciones principales (máximo 8 por archivo)
### H3: Subsecciones (máximo 5 por H2)
#### H4: Detalles específicos (máximo 3 por H3)

<!-- Links externos siempre con target="_blank" -->
[OpenAI API](https://platform.openai.com/docs){:target="_blank"}

<!-- Code blocks siempre con language specification -->
```python
def ejemplo_funcion():
    """Docstring siguiendo Google style."""
    return "resultado"
```

<!-- Alertas consistentes -->
> ⚠️ **Advertencia:** Información crítica
> 💡 **Consejo:** Mejores prácticas
> 📝 **Nota:** Información adicional
> ✅ **Éxito:** Confirmación de logro
```

---

## 💻 Code Quality Standards

### Python Code Style (Basado en Google Style Guide)

#### Imports
```python
# Standard library
import os
import sys
from typing import Dict, List, Optional, Union

# Third-party
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
from src.models import RegresionLineal
from src.utils import cargar_datos
```

#### Function Documentation
```python
def entrenar_modelo(X: np.ndarray, y: np.ndarray, 
                   tasa_aprendizaje: float = 0.01) -> Dict[str, float]:
    """Entrena un modelo de regresión lineal desde cero.
    
    Args:
        X: Matriz de características (n_samples, n_features)
        y: Vector objetivo (n_samples,)
        tasa_aprendizaje: Tasa de aprendizaje para descenso de gradiente
        
    Returns:
        Dict con métricas de entrenamiento:
        - 'mse': Error cuadrático medio
        - 'r2': Coeficiente de determinación
        - 'iteraciones': Número de iteraciones hasta convergencia
        
    Raises:
        ValueError: Si X e y tienen dimensiones incompatibles
        
    Example:
        >>> X = np.random.randn(100, 3)
        >>> y = np.random.randn(100)
        >>> metricas = entrenar_modelo(X, y, tasa_aprendizaje=0.01)
        >>> print(f"R² Score: {metricas['r2']:.3f}")
        R² Score: 0.847
    """
    # Implementación aquí
    pass
```

#### Error Handling
```python
def cargar_dataset(ruta: str) -> pd.DataFrame:
    """Carga dataset con manejo robusto de errores."""
    try:
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Dataset no encontrado: {ruta}")
            
        df = pd.read_csv(ruta)
        
        if df.empty:
            raise ValueError("Dataset está vacío")
            
        return df
        
    except FileNotFoundError as e:
        logger.error(f"Error de archivo: {e}")
        raise
    except pd.errors.EmptyDataError:
        raise ValueError(f"Archivo CSV vacío o malformado: {ruta}")
    except Exception as e:
        logger.error(f"Error inesperado cargando {ruta}: {e}")
        raise
```

### Notebook Standards
```python
# Celda 1: Setup y imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
%matplotlib inline

# Configuración de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

print("✅ Configuración completada")
```

```python
# Celda de título descriptivo
"""
# 🔬 Módulo A - Actividad 2: Implementación de Regresión Lineal

**Objetivo:** Implementar regresión lineal desde cero y comparar con scikit-learn
**Duración:** 45-60 minutos
**Prerrequisitos:** NumPy, conceptos de álgebra lineal

**Entregables:**
1. Función de regresión lineal implementada
2. Comparación de resultados con sklearn
3. Análisis de convergencia del algoritmo
"""
```

---

## 🎨 Visual Design Standards

### Color Palette
```yaml
Primary Colors:
  - Brand Blue: #2563eb
  - Success Green: #10b981  
  - Warning Orange: #f59e0b
  - Error Red: #ef4444
  - Info Purple: #8b5cf6

Neutral Colors:
  - Dark: #1f2937
  - Medium: #6b7280
  - Light: #f3f4f6
  - White: #ffffff

Semantic Colors:
  - Code Background: #f8fafc
  - Note Background: #eff6ff
  - Warning Background: #fffbeb
  - Success Background: #ecfdf5
```

### Typography
```yaml
Headings: 
  - Font: "Inter", system-ui, sans-serif
  - Weights: 600 (semibold) for H1-H2, 500 (medium) for H3-H4

Body Text:
  - Font: "Inter", system-ui, sans-serif  
  - Weight: 400 (regular)
  - Line Height: 1.6

Code:
  - Font: "JetBrains Mono", "Fira Code", monospace
  - Weight: 400 (regular)
  - Line Height: 1.4
```

### Layout Standards
```yaml
Max Content Width: 1200px
Content Padding: 2rem (32px)
Section Spacing: 3rem (48px)
Paragraph Spacing: 1.5rem (24px)

Responsive Breakpoints:
  - Mobile: 640px
  - Tablet: 768px  
  - Desktop: 1024px
  - Wide: 1280px
```

---

## 🧪 Testing Standards

### Code Testing
```python
# test_regresion_lineal.py
import pytest
import numpy as np
from src.models import RegresionLineal

class TestRegresionLineal:
    """Suite de tests comprehensiva para RegresionLineal."""
    
    def setup_method(self):
        """Setup ejecutado antes de cada test."""
        self.modelo = RegresionLineal()
        self.X_simple = np.array([[1], [2], [3], [4], [5]])
        self.y_simple = np.array([2, 4, 6, 8, 10])  # y = 2x
        
    def test_init(self):
        """Test inicialización del modelo."""
        assert self.modelo.theta is None
        assert len(self.modelo.costo_historial) == 0
        
    def test_entrenar_datos_validos(self):
        """Test entrenamiento con datos válidos."""
        self.modelo.entrenar(self.X_simple, self.y_simple)
        
        # Verificar que los parámetros se entrenaron
        assert self.modelo.theta is not None
        assert len(self.modelo.theta) == 2  # intercepto + pendiente
        
        # Verificar convergencia (pendiente cerca de 2, intercepto cerca de 0)
        np.testing.assert_allclose(self.modelo.theta[1], 2.0, rtol=0.1)
        np.testing.assert_allclose(self.modelo.theta[0], 0.0, atol=0.5)
        
    def test_predicciones_precision(self):
        """Test precisión de predicciones."""
        self.modelo.entrenar(self.X_simple, self.y_simple)
        predicciones = self.modelo.predecir(self.X_simple)
        
        # MSE debe ser bajo para datos lineales perfectos
        mse = np.mean((predicciones - self.y_simple) ** 2)
        assert mse < 0.1
        
    def test_datos_invalidos(self):
        """Test manejo de datos inválidos."""
        X_invalido = np.array([[1, 2], [3, 4]])  # 2x2
        y_invalido = np.array([1])  # 1x1 - dimensiones incompatibles
        
        with pytest.raises(ValueError, match="dimensiones incompatibles"):
            self.modelo.entrenar(X_invalido, y_invalido)
            
    @pytest.mark.parametrize("noise_level", [0.1, 0.5, 1.0])
    def test_robustez_ruido(self, noise_level):
        """Test robustez con diferentes niveles de ruido."""
        np.random.seed(42)
        ruido = np.random.normal(0, noise_level, len(self.y_simple))
        y_con_ruido = self.y_simple + ruido
        
        self.modelo.entrenar(self.X_simple, y_con_ruido)
        r2 = self.modelo.puntuar(self.X_simple, y_con_ruido)
        
        # R² debe decrecer con más ruido pero mantenerse razonable
        assert r2 > 0.5 if noise_level <= 0.5 else r2 > 0.1
```

### Documentation Testing
```python
# test_documentation.py
import os
import re
from pathlib import Path

def test_all_readmes_exist():
    """Verifica que todos los módulos tengan README."""
    modulos_path = Path("modulos")
    for modulo_dir in modulos_path.iterdir():
        if modulo_dir.is_dir() and modulo_dir.name.startswith("modulo-"):
            readme_path = modulo_dir / "README.md"
            assert readme_path.exists(), f"Falta README en {modulo_dir}"

def test_link_validity():
    """Verifica que todos los links internos sean válidos."""
    for md_file in Path(".").rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        links = re.findall(r'\[.*?\]\((.*?)\)', content)
        
        for link in links:
            if link.startswith("./") or link.startswith("../"):
                # Link relativo - verificar que el archivo existe
                target_path = (md_file.parent / link).resolve()
                assert target_path.exists(), f"Link roto en {md_file}: {link}"

def test_code_block_syntax():
    """Verifica que todos los code blocks tengan syntax highlighting."""
    for md_file in Path(".").rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        code_blocks = re.findall(r'```(\w*)\n', content)
        
        for i, lang in enumerate(code_blocks):
            assert lang.strip() != "", f"Code block sin lenguaje en {md_file}, bloque {i+1}"
```

---

## 🚀 Performance Standards

### Loading Time Benchmarks
```yaml
Documentation Site:
  - Initial Load: < 2 seconds
  - Page Navigation: < 500ms
  - Search Results: < 300ms
  - Code Syntax Highlighting: < 100ms

Code Execution:
  - Notebook Cell Execution: < 5 seconds per cell
  - Full Module Completion: < 30 minutes
  - Dataset Loading: < 10 seconds for datasets < 100MB
  - Model Training: Progress indicators for > 30 second tasks
```

### Accessibility Standards (WCAG 2.1 AA)
```yaml
Color Contrast: Minimum 4.5:1 ratio
Keyboard Navigation: Full site accessible via keyboard
Screen Reader: Semantic HTML with proper ARIA labels
Font Size: Minimum 16px, scalable to 200%
Images: Alt text for all informational images
Videos: Captions and transcripts provided
```

---

## 📊 Quality Metrics

### Automated Quality Gates
```yaml
Code Quality:
  - Test Coverage: > 85%
  - Linting Score: 10/10 (flake8, black)
  - Type Coverage: > 80% (mypy)
  - Security Scan: 0 high/critical vulnerabilities

Documentation Quality:
  - Link Validation: 100% valid internal links
  - Spell Check: < 5 errors per 1000 words
  - Readability Score: > 60 (Flesch-Kincaid)
  - Image Optimization: < 500KB per image

User Experience:
  - Page Load Speed: > 90 Lighthouse score
  - Mobile Responsiveness: 100% responsive
  - Accessibility: > 95 WAVE score
  - SEO Optimization: > 90 Lighthouse SEO score
```

### Content Review Checklist
```markdown
## Pre-Publication Checklist

### Technical Accuracy
- [ ] Código ejecutado y verificado en ambiente limpio
- [ ] Outputs coinciden con los mostrados en documentación
- [ ] Dependencias y versiones especificadas correctamente
- [ ] Error handling implementado apropiadamente

### Educational Quality  
- [ ] Learning objectives claros y medibles
- [ ] Progresión lógica de conceptos
- [ ] Ejemplos relevantes y prácticos
- [ ] Ejercicios hands-on incluidos
- [ ] Assessment criteria definidos

### Professional Standards
- [ ] Gramática y ortografía revisadas
- [ ] Formato consistente con style guide
- [ ] Enlaces funcionales y actualizados
- [ ] Imágenes optimizadas y con alt text
- [ ] Mobile-responsive verificado

### Industry Alignment
- [ ] Contenido alineado con prácticas actuales de la industria
- [ ] Referencias a herramientas y técnicas estado-del-arte
- [ ] Casos de uso realistas y relevantes
- [ ] Preparación efectiva para entrevistas técnicas
```

---

## 🔄 Continuous Improvement

### Monthly Review Process
```yaml
Content Audit:
  - Actualización de dependencias y versiones
  - Revisión de enlaces externos
  - Feedback de estudiantes integrado
  - Métricas de engagement analizadas

Technology Updates:
  - Nuevas herramientas AI/ML evaluadas
  - Best practices industriales incorporadas
  - Performance optimizations implementadas
  - Security updates aplicados

Community Feedback:
  - Issues de GitHub revisados semanalmente
  - Sugerencias de mejora priorizadas
  - Success stories documentadas
  - Alumni feedback incorporado
```

### Version Control Standards
```yaml
Commit Messages:
  feat: Nueva funcionalidad
  fix: Corrección de bug
  docs: Cambios en documentación
  style: Cambios de formato (sin cambio de lógica)
  refactor: Refactoring de código
  test: Adición o corrección de tests
  chore: Cambios de build, dependencias, etc.

Branch Naming:
  feature/nombre-descriptivo
  hotfix/descripcion-del-fix
  docs/seccion-actualizada
  
Release Versioning:
  MAJOR.MINOR.PATCH (Semantic Versioning)
  Ej: v2.1.3
```

---

**🎯 Objetivo Final:** Cada elemento de Portal 4 debe cumplir o superar los estándares de calidad de las referencias industriales líderes, estableciendo un nuevo benchmark para educación en AI Engineering.
