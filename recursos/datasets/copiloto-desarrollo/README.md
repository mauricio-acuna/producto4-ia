# 💻 Datasets - Copiloto de Desarrollo

## 📋 Descripción

Datasets curados para desarrollar un copiloto de programación con capacidades de generación de código, refactoring, detección de vulnerabilidades y documentación automática.

## 📁 Datasets Incluidos

### 1. 🐍 Python Code Repository
**Archivo:** `python_functions_dataset.json`  
**Descripción:** 10,000 funciones Python con documentación y metadatos  
**Uso:** Entrenamiento de embeddings de código y generación contextual

```json
{
  "function_id": "py_func_001",
  "code": "def calculate_fibonacci(n: int) -> int:\n    \"\"\"Calculate nth Fibonacci number using iterative approach.\"\"\"\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b",
  "docstring": "Calculate nth Fibonacci number using iterative approach.",
  "complexity": "O(n)",
  "category": "algorithms",
  "imports": [],
  "test_cases": [
    {"input": 0, "output": 0},
    {"input": 1, "output": 1},
    {"input": 10, "output": 55}
  ],
  "quality_score": 9.2,
  "common_bugs": ["off_by_one_error", "negative_input_handling"]
}
```

### 2. 🔍 Code Vulnerabilities Database
**Archivo:** `security_vulnerabilities.json`  
**Descripción:** 2,500 ejemplos de código vulnerable con fixes  
**Uso:** Entrenamiento de detección de vulnerabilidades

```json
{
  "vuln_id": "sql_injection_001",
  "vulnerable_code": "def get_user(user_id):\n    query = f\"SELECT * FROM users WHERE id = {user_id}\"\n    return execute_query(query)",
  "vulnerability_type": "sql_injection",
  "severity": "high",
  "cwe_id": "CWE-89",
  "fixed_code": "def get_user(user_id: int) -> Optional[Dict]:\n    query = \"SELECT * FROM users WHERE id = %s\"\n    return execute_query(query, (user_id,))",
  "explanation": "Use parameterized queries to prevent SQL injection attacks",
  "owasp_category": "A03:2021 - Injection",
  "detection_patterns": ["string_formatting_in_sql", "f_string_in_query"]
}
```

### 3. 📚 API Documentation Corpus
**Archivo:** `api_documentation.json`  
**Descripción:** Documentación de 500+ APIs populares  
**Uso:** Generación automática de documentación

```json
{
  "api_id": "fastapi_endpoint_001",
  "framework": "FastAPI",
  "endpoint_code": "@app.post(\"/users/\")\nasync def create_user(user: UserCreate) -> User:\n    \"\"\"Create a new user account.\"\"\"\n    return await user_service.create(user)",
  "generated_docs": {
    "summary": "Create a new user account",
    "parameters": [
      {
        "name": "user",
        "type": "UserCreate",
        "required": true,
        "description": "User data for account creation"
      }
    ],
    "responses": {
      "200": {"description": "User created successfully", "model": "User"},
      "400": {"description": "Invalid user data"},
      "409": {"description": "User already exists"}
    }
  }
}
```

### 4. 🔄 Refactoring Patterns
**Archivo:** `refactoring_examples.json`  
**Descripción:** 1,000 ejemplos de código antes/después de refactoring  
**Uso:** Sugerencias automáticas de mejoras

```json
{
  "refactor_id": "extract_method_001",
  "before_code": "def process_order(order):\n    # Validate order\n    if not order.items:\n        raise ValueError(\"Empty order\")\n    if order.total < 0:\n        raise ValueError(\"Negative total\")\n    \n    # Calculate tax\n    tax_rate = 0.08 if order.state == 'CA' else 0.05\n    tax = order.subtotal * tax_rate\n    \n    # Process payment\n    payment_result = payment_gateway.charge(\n        amount=order.total + tax,\n        card=order.payment_method\n    )\n    \n    return payment_result",
  "after_code": "def process_order(order):\n    validate_order(order)\n    tax = calculate_tax(order)\n    return process_payment(order, tax)\n\ndef validate_order(order):\n    if not order.items:\n        raise ValueError(\"Empty order\")\n    if order.total < 0:\n        raise ValueError(\"Negative total\")\n\ndef calculate_tax(order):\n    tax_rate = 0.08 if order.state == 'CA' else 0.05\n    return order.subtotal * tax_rate\n\ndef process_payment(order, tax):\n    return payment_gateway.charge(\n        amount=order.total + tax,\n        card=order.payment_method\n    )",
  "refactoring_type": "extract_method",
  "benefits": ["improved_readability", "better_testability", "single_responsibility"],
  "complexity_before": 8,
  "complexity_after": 3
}
```

### 5. 🐛 GitHub Issues Dataset
**Archivo:** `github_issues.json`  
**Descripción:** 5,000 issues reales con soluciones  
**Uso:** Contextualización de problemas comunes

```json
{
  "issue_id": "gh_issue_001",
  "title": "TypeError: 'NoneType' object is not subscriptable",
  "description": "Obteniendo este error al intentar acceder a datos de usuario: `user_data['email']` devuelve TypeError",
  "code_context": "def get_user_email(user_id):\n    user_data = get_user_from_db(user_id)\n    return user_data['email']  # Error here",
  "solution": "def get_user_email(user_id):\n    user_data = get_user_from_db(user_id)\n    if user_data is None:\n        return None\n    return user_data.get('email')",
  "explanation": "Always check for None before accessing dictionary keys",
  "tags": ["python", "error_handling", "none_type"],
  "language": "python",
  "difficulty": "beginner"
}
```

## 🎯 Casos de Uso por Dataset

### Generación de Código
- **Python Functions:** Base para autocompletado contextual
- **API Documentation:** Generación de endpoints RESTful
- **Refactoring Patterns:** Sugerencias de mejoras automáticas

### Detección de Problemas
- **Vulnerabilities Database:** Scanning de seguridad
- **GitHub Issues:** Identificación de errores comunes
- **Refactoring Examples:** Code smell detection

### Documentación Automática
- **API Documentation:** Templates de documentación
- **Python Functions:** Generación de docstrings
- **GitHub Issues:** FAQs automáticas

## 📊 Métricas de Evaluación

### Precisión de Generación
```python
# Métrica: Code similarity (AST-based)
def evaluate_code_generation(predicted_code, expected_code):
    return {
        'syntax_correctness': check_syntax(predicted_code),
        'semantic_similarity': ast_similarity(predicted_code, expected_code),
        'functionality_match': test_execution_match(predicted_code, expected_code)
    }
```

### Detección de Vulnerabilidades
```python
# Métrica: Security detection accuracy
def evaluate_vulnerability_detection(predictions, ground_truth):
    return {
        'precision': tp / (tp + fp),
        'recall': tp / (tp + fn),
        'f1_score': 2 * (precision * recall) / (precision + recall),
        'false_positive_rate': fp / (fp + tn)
    }
```

## 🚀 Uso Recomendado

### Semana 1: MVP Básico
- Usar 10% de Python Functions para embeddings básicos
- Implementar detección simple con 20 vulnerabilidades top
- Generar documentación básica con templates

### Semana 2: Features Avanzadas  
- Expandir a 50% del dataset de funciones
- Agregar refactoring suggestions con patterns básicos
- Integrar GitHub issues para contexto

### Semana 3: Optimización
- Usar dataset completo para embeddings
- Implementar detección avanzada de vulnerabilidades
- Refactoring inteligente con análisis de complejidad

### Semana 4: Refinamiento
- Fine-tuning con casos específicos del dominio
- Optimización de precisión y recall
- Benchmarking contra herramientas existentes

---

**📥 Descarga:** Los datasets están disponibles en formato JSON comprimido en la carpeta correspondiente.
