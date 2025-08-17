# 🎯 Scripts de Benchmark Portal 4

Esta carpeta contiene scripts automatizados para evaluar y comparar diferentes sistemas de IA desarrollados en los capstones de Portal 4.

## 🎯 Objetivo

Proporcionar herramientas estandarizadas para medir el rendimiento, precisión y eficiencia de:
- Sistemas de retrieval (BM25, semántico, híbrido)
- Sistemas RAG completos
- Agentes de IA especializados
- Métricas de evaluación

## � Scripts Disponibles

### 1. `benchmark_orchestrator.py` ⭐ **PRINCIPAL**
Script maestro que coordina todos los benchmarks del sistema.

**Funcionalidades:**
- Ejecuta suite completo de benchmarks
- Genera reportes consolidados
- Maneja configuración centralizada
- Exporta resultados en múltiples formatos

**Uso:**
```bash
python benchmark_orchestrator.py
```

### 2. `retrieval_benchmark.py`
Evalúa y compara estrategias de recuperación de información.

**Métricas:**
- Precision@K (1, 5)
- Recall@K (5)
- NDCG@K (5)
- Tiempo de respuesta promedio

**Métodos evaluados:**
- BM25 (baseline)
- Búsqueda semántica
- Híbrido (BM25 + semántica)

**Uso:**
```bash
python retrieval_benchmark.py
```

### 3. `rag_benchmark.py`
Benchmark completo para sistemas RAG (Retrieval + Generation).

**Métricas evaluadas:**
- **Retrieval:** Precision@5, relevancia de contexto
- **Generation:** Relevancia de respuesta, faithfulness, similitud
- **Performance:** Tiempos de retrieval, generación y total

**Configuraciones evaluadas:**
- BM25 + GPT-3.5
- Semántico + GPT-3.5  
- Híbrido + GPT-3.5
- Híbrido + GPT-4
- Híbrido + Modelo Local

**Uso:**
```bash
python rag_benchmark.py
```

### 4. `agent_benchmark.py`
Evaluación de agentes de IA en tareas específicas por tipo.

**Tipos de tareas:**
- Generación de código
- Análisis de documentos
- Resolución de problemas
- Toma de decisiones

**Métricas:**
- Tasa de éxito general
- Precisión promedio por agente
- Eficiencia temporal
- Performance específica por tipo de tarea

**Agentes evaluados:**
- Dev Copilot Agent
- Enterprise Assistant Agent
- Legal Finance Agent  
- Analytics Agent
- Generic AI Agent

**Uso:**
```bash
python agent_benchmark.py
```

### 5. `metrics_calculator.py`
Biblioteca de métricas y funciones de evaluación (sin dependencias externas).

**Métricas incluidas:**
- **Retrieval:** Precision@K, Recall@K, NDCG@K, MRR
- **Text Generation:** BLEU, ROUGE-L, similitud coseno
- **Classification:** Accuracy, F1-score
- **Regression:** MAE, MSE, RMSE
- **Similarity:** Jaccard, coseno, Levenshtein

**Uso como librería:**
```python
from metrics_calculator import MetricsCalculator
calc = MetricsCalculator()
precision = calc.precision_at_k(retrieved, relevant, 5)
```

### 💰 Análisis de Costos
**Archivo:** `cost_analysis.py`  
**Uso:** Tracking de costos por token y por query  
**Métricas:** Cost/query, Cost/token, Cost efficiency ratio

### ⏱️ Performance Testing
**Archivo:** `latency_benchmark.py`  
**Uso:** Medición de latencia y throughput  
**Métricas:** P50, P95, P99 latency, QPS, Concurrent users

### 🎯 Accuracy Testing
**Archivo:** `accuracy_benchmark.py`  
**Uso:** Evaluación vs. ground truth datasets  
**Métricas:** Accuracy, F1, BLEU, ROUGE (según capstone)

### 📊 End-to-End Evaluation
**Archivo:** `e2e_benchmark.py`  
**Uso:** Evaluación integral del sistema completo  
**Métricas:** Business metrics, User satisfaction, System reliability

## 🚀 Configuración Rápida

### Instalación de Dependencias

```bash
pip install -r benchmark_requirements.txt
```

### Configuración Básica

```python
# benchmark_config.py
BENCHMARK_CONFIG = {
    "models": {
        "primary": "gpt-4-turbo",
        "baseline": "gpt-3.5-turbo"
    },
    "datasets": {
        "test_size": 1000,
        "validation_split": 0.2
    },
    "metrics": {
        "latency_threshold": 5.0,  # seconds
        "accuracy_threshold": 0.85,
        "cost_threshold": 0.10     # $ per query
    }
}
```

## 📊 Métricas por Tipo de Capstone

### 💻 Copiloto de Desarrollo
```python
COPILOT_METRICS = {
    "code_generation": ["syntax_correctness", "semantic_similarity", "execution_success"],
    "vulnerability_detection": ["precision", "recall", "f1_score", "false_positive_rate"],
    "performance": ["response_time", "throughput", "memory_usage"],
    "user_experience": ["acceptance_rate", "time_to_completion", "satisfaction_score"]
}
```

### 🏢 Asistente Empresarial
```python
ENTERPRISE_METRICS = {
    "qa_accuracy": ["answer_correctness", "source_citation", "completeness"],
    "access_control": ["rbac_compliance", "unauthorized_access_rate", "audit_completeness"],
    "business_impact": ["query_resolution_rate", "support_ticket_reduction", "user_adoption"],
    "compliance": ["gdpr_compliance", "audit_readiness", "security_score"]
}
```

### ⚖️ Legal/Finanzas
```python
LEGAL_METRICS = {
    "contract_analysis": ["clause_extraction_f1", "risk_assessment_accuracy", "citation_precision"],
    "jurisprudence": ["case_relevance", "holding_extraction", "precedent_accuracy"],
    "compliance": ["regulatory_adherence", "audit_trail_completeness", "risk_scoring_accuracy"],
    "precision": ["false_positive_rate", "confidence_calibration", "expert_agreement"]
}
```

### 📊 Analytics Agent
```python
ANALYTICS_METRICS = {
    "insight_generation": ["insight_accuracy", "business_relevance", "actionability_score"],
    "forecasting": ["mape", "directional_accuracy", "confidence_intervals"],
    "visualization": ["chart_appropriateness", "data_accuracy", "interpretability"],
    "automation": ["end_to_end_time", "manual_intervention_rate", "error_rate"]
}
```

## 🔧 Scripts de Uso

### 1. Benchmark de Retrieval

```python
# Ejemplo de uso: retrieval_benchmark.py
from benchmark_suite import RetrievalBenchmark

benchmark = RetrievalBenchmark(
    dataset_path="datasets/qa_test_set.json",
    models=["bm25", "semantic", "hybrid"]
)

results = benchmark.run_evaluation()
benchmark.generate_report(results, output_path="reports/retrieval_results.html")
```

**Output esperado:**
```
=== Retrieval Benchmark Results ===
Dataset: QA Test Set (1000 queries)

BM25 Baseline:
  Precision@1: 0.65
  Precision@5: 0.78
  Recall@5: 0.82
  NDCG@5: 0.71

Semantic Search:
  Precision@1: 0.78 (+20%)
  Precision@5: 0.84 (+8%)
  Recall@5: 0.89 (+9%)
  NDCG@5: 0.81 (+14%)

Hybrid (Weighted):
  Precision@1: 0.83 (+28%)
  Precision@5: 0.87 (+12%)
  Recall@5: 0.91 (+11%)
  NDCG@5: 0.85 (+20%)

Winner: Hybrid approach with 20% improvement in relevance
```

### 2. Análisis de Costos

```python
# Ejemplo de uso: cost_analysis.py
from benchmark_suite import CostAnalyzer

analyzer = CostAnalyzer(
    api_logs="logs/api_usage.json",
    pricing_config="config/model_pricing.yaml"
)

cost_report = analyzer.analyze_costs(
    time_range="last_30_days",
    breakdown_by=["model", "user_type", "feature"]
)

analyzer.generate_cost_optimization_recommendations(cost_report)
```

**Output esperado:**
```
=== Cost Analysis Report ===
Period: Last 30 days
Total Queries: 25,000

Cost Breakdown:
  GPT-4 Turbo: $1,250 (50k tokens avg/query)
  GPT-3.5 Turbo: $320 (backup queries)
  Embedding: $45 (vector generation)
  Total: $1,615

Cost per Query: $0.065
Cost per User: $12.50/month (avg 200 queries/user)

Optimization Opportunities:
  1. Use GPT-3.5 for simple queries: -30% cost
  2. Implement caching: -15% cost  
  3. Optimize prompts: -10% token usage
  Potential savings: $500/month (31%)
```

### 3. Latency Testing

```python
# Ejemplo de uso: latency_benchmark.py
from benchmark_suite import LatencyBenchmark

benchmark = LatencyBenchmark(
    endpoint="https://api.your-capstone.com",
    test_scenarios="scenarios/load_test.yaml"
)

# Test con carga creciente
load_results = benchmark.run_load_test(
    users=[1, 5, 10, 25, 50],
    duration_minutes=5
)

benchmark.generate_performance_report(load_results)
```

**Output esperado:**
```
=== Latency Benchmark Results ===

Single User Performance:
  Average Response Time: 1.8s
  P95 Response Time: 3.2s
  P99 Response Time: 5.1s
  Success Rate: 99.8%

Load Testing Results:
  1 user:  1.8s avg, 100% success
  5 users: 2.1s avg, 100% success  
  10 users: 2.8s avg, 99.5% success
  25 users: 4.2s avg, 98.1% success
  50 users: 7.8s avg, 92.3% success ⚠️

Bottleneck: Vector DB queries under high load
Recommendation: Implement connection pooling
```

## 📈 Reportes Automatizados

### Dashboard HTML
```python
# Genera dashboard interactivo
python scripts/generate_dashboard.py --capstone-type copilot-dev
```

### Reporte PDF
```python
# Genera reporte ejecutivo
python scripts/generate_executive_report.py --format pdf --include-graphs
```

### CI/CD Integration
```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark
on:
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Benchmarks
        run: |
          python scripts/e2e_benchmark.py --regression-test
          python scripts/cost_analysis.py --budget-check
      - name: Comment PR
        run: |
          python scripts/post_benchmark_results.py --pr-number ${{ github.event.number }}
```

## 🎯 Benchmarks por Semana

### Semana 1: Baseline
```bash
# Establecer baseline con MVP
python scripts/baseline_benchmark.py --save-baseline week1
```

### Semana 2: Feature Development
```bash
# Comparar vs baseline
python scripts/regression_test.py --compare-to week1
```

### Semana 3: Optimization
```bash
# Performance tuning validation
python scripts/optimization_benchmark.py --focus latency,cost
```

### Semana 4: Final Evaluation
```bash
# Comprehensive final benchmark
python scripts/final_benchmark.py --generate-portfolio-report
```

## 📋 Checklist de Benchmark

### Antes de Benchmark
- [ ] Datos de test listos y validados
- [ ] Ambiente de testing configurado
- [ ] Baseline establecido para comparación
- [ ] Métricas de éxito definidas

### Durante Benchmark
- [ ] Multiple runs para consistencia
- [ ] Monitoring de recursos del sistema
- [ ] Logging detallado habilitado
- [ ] Conditions normalizadas (misma infraestructura)

### Después de Benchmark
- [ ] Resultados documentados
- [ ] Gráficos y visualizaciones generados
- [ ] Comparación vs. objectives
- [ ] Recommendations para optimización

---

**💡 Consejo:** Ejecuta benchmarks regularmente durante el desarrollo para identificar regresiones temprano y optimizar basado en datos objetivos.
