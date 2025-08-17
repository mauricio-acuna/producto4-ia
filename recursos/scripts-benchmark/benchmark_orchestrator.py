#!/usr/bin/env python3
"""
Portal 4 Benchmark Suite - Orchestrator Principal
Ejecuta y coordina todos los benchmarks del sistema
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Importar módulos de benchmark (simulados si no están disponibles)
try:
    from .retrieval_benchmark import RetrievalBenchmark
    from .rag_benchmark import RAGBenchmark  
    from .agent_benchmark import AgentBenchmark
    from .metrics_calculator import MetricsCalculator, BenchmarkEvaluator
except ImportError:
    print("⚠️  Running in standalone mode - benchmark modules not found")
    RetrievalBenchmark = None
    RAGBenchmark = None
    AgentBenchmark = None
    MetricsCalculator = None
    BenchmarkEvaluator = None

class Portal4BenchmarkSuite:
    """Suite principal de benchmarks para Portal 4"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.results_dir = Path("benchmark_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Timestamp para esta ejecución
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Resultados consolidados
        self.all_results = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Carga configuración de benchmarks"""
        default_config = {
            "benchmarks": {
                "retrieval": {
                    "enabled": True,
                    "methods": ["bm25", "semantic", "hybrid"],
                    "dataset": "sample_test_dataset.json"
                },
                "rag": {
                    "enabled": True, 
                    "systems": [
                        ("BM25 + GPT-3.5", "bm25", "gpt-3.5"),
                        ("Hybrid + GPT-4", "hybrid", "gpt-4")
                    ],
                    "dataset": "rag_test_dataset.json"
                },
                "agents": {
                    "enabled": True,
                    "agents": [
                        "Dev Copilot Agent",
                        "Enterprise Assistant Agent", 
                        "Analytics Agent"
                    ],
                    "tasks": "agent_benchmark_tasks.json"
                }
            },
            "output": {
                "generate_html_reports": True,
                "generate_json_summary": True,
                "generate_csv_export": True
            },
            "performance": {
                "max_parallel_tests": 3,
                "timeout_per_test": 300
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                # Merge configurations
                default_config.update(user_config)
        
        return default_config
    
    def run_retrieval_benchmark(self) -> Dict[str, Any]:
        """Ejecuta benchmark de retrieval"""
        print("\n🔍 Starting Retrieval Benchmark...")
        
        if not RetrievalBenchmark:
            print("⚠️  Retrieval benchmark module not available - creating mock results")
            return self._create_mock_retrieval_results()
        
        try:
            config = self.config["benchmarks"]["retrieval"]
            dataset_path = config["dataset"]
            
            # Crear dataset si no existe
            if not Path(dataset_path).exists():
                self._create_sample_retrieval_dataset(dataset_path)
            
            benchmark = RetrievalBenchmark(dataset_path, {})
            results = benchmark.run_full_benchmark()
            
            # Generar reporte
            report_path = benchmark.generate_report(results, 
                f"benchmark_results/retrieval_report_{self.timestamp}.html")
            
            return {
                "type": "retrieval",
                "timestamp": self.timestamp,
                "results": {name: {
                    "precision_at_1": result.precision_at_1,
                    "precision_at_5": result.precision_at_5,
                    "recall_at_5": result.recall_at_5,
                    "ndcg_at_5": result.ndcg_at_5,
                    "avg_response_time": result.avg_response_time,
                    "total_queries": result.total_queries
                } for name, result in results.items()},
                "report_path": report_path,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ Error in retrieval benchmark: {e}")
            return {"type": "retrieval", "status": "failed", "error": str(e)}
    
    def run_rag_benchmark(self) -> Dict[str, Any]:
        """Ejecuta benchmark de RAG"""
        print("\n🤖 Starting RAG Benchmark...")
        
        if not RAGBenchmark:
            print("⚠️  RAG benchmark module not available - creating mock results")
            return self._create_mock_rag_results()
        
        try:
            config = self.config["benchmarks"]["rag"]
            dataset_path = config["dataset"]
            
            # Crear dataset si no existe
            if not Path(dataset_path).exists():
                self._create_sample_rag_dataset(dataset_path)
            
            benchmark = RAGBenchmark(dataset_path, {})
            results = benchmark.run_comparative_benchmark()
            
            # Generar reporte
            report_path = benchmark.generate_detailed_report(results, 
                f"benchmark_results/rag_report_{self.timestamp}.html")
            
            return {
                "type": "rag",
                "timestamp": self.timestamp,
                "results": {name: {
                    "retrieval_precision_at_5": result.retrieval_precision_at_5,
                    "context_relevance": result.context_relevance,
                    "answer_relevance": result.answer_relevance,
                    "faithfulness": result.faithfulness,
                    "answer_similarity": result.answer_similarity,
                    "avg_total_time": result.avg_total_time,
                    "success_rate": result.successful_queries / result.total_queries
                } for name, result in results.items()},
                "report_path": report_path,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ Error in RAG benchmark: {e}")
            return {"type": "rag", "status": "failed", "error": str(e)}
    
    def run_agent_benchmark(self) -> Dict[str, Any]:
        """Ejecuta benchmark de agentes"""
        print("\n🎯 Starting Agent Benchmark...")
        
        if not AgentBenchmark:
            print("⚠️  Agent benchmark module not available - creating mock results")
            return self._create_mock_agent_results()
        
        try:
            config = self.config["benchmarks"]["agents"]
            tasks_path = config["tasks"]
            
            # Crear tasks si no existen
            if not Path(tasks_path).exists():
                self._create_sample_agent_tasks(tasks_path)
            
            benchmark = AgentBenchmark(tasks_path)
            agents = config["agents"]
            results = benchmark.run_comparative_benchmark(agents)
            
            # Generar reporte
            report_path = benchmark.generate_report(results, 
                f"benchmark_results/agent_report_{self.timestamp}.html")
            
            return {
                "type": "agents",
                "timestamp": self.timestamp,
                "results": {name: {
                    "success_rate": result.success_rate,
                    "avg_accuracy": result.avg_accuracy,
                    "avg_efficiency": result.avg_efficiency,
                    "avg_execution_time": result.avg_execution_time,
                    "successful_tasks": result.successful_tasks,
                    "total_tasks": result.total_tasks
                } for name, result in results.items()},
                "report_path": report_path,
                "status": "completed"
            }
            
        except Exception as e:
            print(f"❌ Error in agent benchmark: {e}")
            return {"type": "agents", "status": "failed", "error": str(e)}
    
    def run_full_benchmark_suite(self) -> Dict[str, Any]:
        """Ejecuta suite completo de benchmarks"""
        print("🚀 Portal 4 - Full Benchmark Suite")
        print("="*60)
        
        start_time = time.time()
        suite_results = {
            "suite_id": f"portal4_benchmark_{self.timestamp}",
            "start_time": datetime.now().isoformat(),
            "config": self.config,
            "benchmarks": {}
        }
        
        # Ejecutar benchmarks según configuración
        if self.config["benchmarks"]["retrieval"]["enabled"]:
            suite_results["benchmarks"]["retrieval"] = self.run_retrieval_benchmark()
        
        if self.config["benchmarks"]["rag"]["enabled"]:
            suite_results["benchmarks"]["rag"] = self.run_rag_benchmark()
        
        if self.config["benchmarks"]["agents"]["enabled"]:
            suite_results["benchmarks"]["agents"] = self.run_agent_benchmark()
        
        # Finalizar
        total_time = time.time() - start_time
        suite_results["end_time"] = datetime.now().isoformat()
        suite_results["total_duration"] = total_time
        
        # Guardar resultados
        self._save_consolidated_results(suite_results)
        
        # Generar reporte consolidado
        self._generate_consolidated_report(suite_results)
        
        print(f"\n✅ Benchmark suite completed in {total_time:.2f} seconds")
        return suite_results
    
    def _save_consolidated_results(self, results: Dict[str, Any]):
        """Guarda resultados consolidados"""
        # JSON completo
        json_path = self.results_dir / f"consolidated_results_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"📄 Results saved: {json_path}")
        
        # CSV summary si está habilitado
        if self.config["output"]["generate_csv_export"]:
            self._export_to_csv(results)
    
    def _export_to_csv(self, results: Dict[str, Any]):
        """Exporta resultados a CSV"""
        csv_path = self.results_dir / f"benchmark_summary_{self.timestamp}.csv"
        
        with open(csv_path, 'w') as f:
            f.write("benchmark_type,system_name,metric_name,metric_value\n")
            
            for bench_type, bench_data in results["benchmarks"].items():
                if "results" in bench_data:
                    for system_name, metrics in bench_data["results"].items():
                        for metric_name, metric_value in metrics.items():
                            f.write(f"{bench_type},{system_name},{metric_name},{metric_value}\n")
        
        print(f"📊 CSV export: {csv_path}")
    
    def _generate_consolidated_report(self, results: Dict[str, Any]):
        """Genera reporte HTML consolidado"""
        if not self.config["output"]["generate_html_reports"]:
            return
        
        # Encontrar mejores sistemas por categoría
        best_systems = self._identify_best_systems(results)
        
        # Crear tabla de resumen
        summary_table = self._create_summary_table(results)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Portal 4 - Benchmark Suite Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
                .section {{ background: #f8f9fa; padding: 20px; margin: 20px 0; border-radius: 8px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                th {{ background-color: #e9ecef; font-weight: bold; }}
                .winner {{ background-color: #d4edda; color: #155724; font-weight: bold; }}
                .status-completed {{ color: #28a745; font-weight: bold; }}
                .status-failed {{ color: #dc3545; font-weight: bold; }}
                .metric-box {{ background: white; padding: 15px; margin: 10px; border-radius: 5px; 
                             border-left: 4px solid #007bff; }}
                .timestamp {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Portal 4 - Benchmark Suite Report</h1>
                <p>Comprehensive AI System Evaluation</p>
                <p class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p class="timestamp">Suite ID: {results['suite_id']}</p>
            </div>
            
            <div class="section">
                <h2>📊 Executive Summary</h2>
                <p><strong>Total Duration:</strong> {results.get('total_duration', 0):.2f} seconds</p>
                <p><strong>Benchmarks Executed:</strong> {len([b for b in results['benchmarks'].values() if b.get('status') == 'completed'])}</p>
                <p><strong>Status:</strong> {self._get_overall_status(results)}</p>
                
                <h3>🏆 Best Performing Systems</h3>
                {self._format_best_systems(best_systems)}
            </div>
            
            <div class="section">
                <h2>📈 Detailed Results</h2>
                {summary_table}
            </div>
            
            <div class="section">
                <h2>📋 Individual Reports</h2>
                <ul>
                {self._format_individual_reports(results)}
                </ul>
            </div>
            
            <div class="section">
                <h2>💡 Recommendations</h2>
                {self._generate_recommendations(results)}
            </div>
            
            <div class="section">
                <h2>🔧 Configuration Used</h2>
                <pre style="background: #f1f3f4; padding: 15px; border-radius: 5px; overflow-x: auto;">
{json.dumps(results['config'], indent=2)}
                </pre>
            </div>
        </body>
        </html>
        """
        
        report_path = self.results_dir / f"consolidated_report_{self.timestamp}.html"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"📊 Consolidated report: {report_path}")
    
    def _identify_best_systems(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Identifica mejores sistemas por categoría"""
        best = {}
        
        for bench_type, bench_data in results["benchmarks"].items():
            if bench_data.get("status") == "completed" and "results" in bench_data:
                systems = bench_data["results"]
                
                if bench_type == "retrieval":
                    # Mejor por NDCG@5
                    best_system = max(systems.items(), key=lambda x: x[1].get("ndcg_at_5", 0))
                    best[f"{bench_type}_best"] = best_system[0]
                
                elif bench_type == "rag":
                    # Mejor por answer_relevance
                    best_system = max(systems.items(), key=lambda x: x[1].get("answer_relevance", 0))
                    best[f"{bench_type}_best"] = best_system[0]
                
                elif bench_type == "agents":
                    # Mejor por accuracy
                    best_system = max(systems.items(), key=lambda x: x[1].get("avg_accuracy", 0))
                    best[f"{bench_type}_best"] = best_system[0]
        
        return best
    
    def _create_summary_table(self, results: Dict[str, Any]) -> str:
        """Crea tabla de resumen de resultados"""
        table_html = ""
        
        for bench_type, bench_data in results["benchmarks"].items():
            status_class = "status-completed" if bench_data.get("status") == "completed" else "status-failed"
            
            table_html += f"""
            <h3>{bench_type.title()} Benchmark</h3>
            <p class="{status_class}">Status: {bench_data.get('status', 'unknown').title()}</p>
            """
            
            if bench_data.get("status") == "completed" and "results" in bench_data:
                table_html += "<table><thead><tr><th>System</th><th>Key Metrics</th></tr></thead><tbody>"
                
                for system_name, metrics in bench_data["results"].items():
                    key_metrics = self._format_key_metrics(bench_type, metrics)
                    table_html += f"<tr><td>{system_name}</td><td>{key_metrics}</td></tr>"
                
                table_html += "</tbody></table>"
            
            elif bench_data.get("status") == "failed":
                table_html += f"<p style='color: #dc3545;'>Error: {bench_data.get('error', 'Unknown error')}</p>"
        
        return table_html
    
    def _format_key_metrics(self, bench_type: str, metrics: Dict[str, float]) -> str:
        """Formatea métricas clave por tipo de benchmark"""
        if bench_type == "retrieval":
            return f"P@5: {metrics.get('precision_at_5', 0):.3f}, NDCG@5: {metrics.get('ndcg_at_5', 0):.3f}"
        elif bench_type == "rag":
            return f"Answer Rel: {metrics.get('answer_relevance', 0):.3f}, Faithfulness: {metrics.get('faithfulness', 0):.3f}"
        elif bench_type == "agents":
            return f"Accuracy: {metrics.get('avg_accuracy', 0):.3f}, Success: {metrics.get('success_rate', 0):.3f}"
        else:
            return "N/A"
    
    def _format_best_systems(self, best_systems: Dict[str, str]) -> str:
        """Formatea mejores sistemas"""
        if not best_systems:
            return "<p>No best systems identified</p>"
        
        html = "<ul>"
        for category, system in best_systems.items():
            category_name = category.replace("_best", "").title()
            html += f"<li><strong>{category_name}:</strong> {system}</li>"
        html += "</ul>"
        return html
    
    def _format_individual_reports(self, results: Dict[str, Any]) -> str:
        """Formatea enlaces a reportes individuales"""
        html = ""
        for bench_type, bench_data in results["benchmarks"].items():
            if "report_path" in bench_data:
                html += f"<li><a href='{bench_data['report_path']}'>{bench_type.title()} Detailed Report</a></li>"
        return html
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> str:
        """Genera recomendaciones basadas en resultados"""
        recommendations = []
        
        for bench_type, bench_data in results["benchmarks"].items():
            if bench_data.get("status") == "completed":
                if bench_type == "retrieval":
                    recommendations.append("✓ Retrieval systems evaluated - consider hybrid approaches for best results")
                elif bench_type == "rag":
                    recommendations.append("✓ RAG systems evaluated - balance accuracy with response time")
                elif bench_type == "agents":
                    recommendations.append("✓ Agent systems evaluated - specialize agents for specific task types")
        
        if not recommendations:
            recommendations.append("• Complete benchmark execution to receive recommendations")
        
        return "<ul>" + "".join(f"<li>{rec}</li>" for rec in recommendations) + "</ul>"
    
    def _get_overall_status(self, results: Dict[str, Any]) -> str:
        """Determina status general del suite"""
        statuses = [bench.get("status") for bench in results["benchmarks"].values()]
        
        if all(status == "completed" for status in statuses):
            return "✅ All benchmarks completed successfully"
        elif any(status == "completed" for status in statuses):
            return "⚠️ Partially completed"
        else:
            return "❌ Failed to complete benchmarks"
    
    # Mock methods para cuando los módulos no están disponibles
    def _create_mock_retrieval_results(self) -> Dict[str, Any]:
        """Crea resultados mock para retrieval"""
        return {
            "type": "retrieval",
            "timestamp": self.timestamp,
            "results": {
                "BM25": {"precision_at_5": 0.65, "ndcg_at_5": 0.72, "avg_response_time": 0.15},
                "Semantic": {"precision_at_5": 0.78, "ndcg_at_5": 0.81, "avg_response_time": 0.45},
                "Hybrid": {"precision_at_5": 0.82, "ndcg_at_5": 0.86, "avg_response_time": 0.35}
            },
            "status": "completed (simulated)"
        }
    
    def _create_mock_rag_results(self) -> Dict[str, Any]:
        """Crea resultados mock para RAG"""
        return {
            "type": "rag", 
            "timestamp": self.timestamp,
            "results": {
                "BM25 + GPT-3.5": {"answer_relevance": 0.75, "faithfulness": 0.68, "avg_total_time": 1.2},
                "Hybrid + GPT-4": {"answer_relevance": 0.89, "faithfulness": 0.84, "avg_total_time": 2.1}
            },
            "status": "completed (simulated)"
        }
    
    def _create_mock_agent_results(self) -> Dict[str, Any]:
        """Crea resultados mock para agentes"""
        return {
            "type": "agents",
            "timestamp": self.timestamp, 
            "results": {
                "Dev Copilot Agent": {"avg_accuracy": 0.87, "success_rate": 0.93, "avg_execution_time": 2.5},
                "Enterprise Assistant": {"avg_accuracy": 0.81, "success_rate": 0.89, "avg_execution_time": 1.8},
                "Analytics Agent": {"avg_accuracy": 0.79, "success_rate": 0.85, "avg_execution_time": 3.2}
            },
            "status": "completed (simulated)"
        }
    
    def _create_sample_retrieval_dataset(self, path: str):
        """Crea dataset de ejemplo para retrieval"""
        sample_data = [
            {"id": "q001", "query": "vacation policy", "relevant_docs": ["doc_1", "doc_3"]},
            {"id": "q002", "query": "remote work", "relevant_docs": ["doc_2", "doc_5"]},
            {"id": "q003", "query": "security requirements", "relevant_docs": ["doc_4", "doc_6"]}
        ]
        
        with open(path, 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    def _create_sample_rag_dataset(self, path: str):
        """Crea dataset de ejemplo para RAG"""
        sample_data = [
            {
                "id": "rag_001",
                "query": "How many vacation days?",
                "expected_answer": "15 days for new employees",
                "relevant_contexts": ["vacation policy document"]
            }
        ]
        
        with open(path, 'w') as f:
            json.dump(sample_data, f, indent=2)
    
    def _create_sample_agent_tasks(self, path: str):
        """Crea tasks de ejemplo para agentes"""
        sample_tasks = [
            {
                "id": "task_001",
                "type": "code_generation", 
                "description": "Generate Python function",
                "input": {"language": "python"},
                "expected_output": {"code": "def example(): pass"},
                "evaluation_criteria": ["correctness"]
            }
        ]
        
        with open(path, 'w') as f:
            json.dump(sample_tasks, f, indent=2)

def create_default_config():
    """Crea archivo de configuración por defecto"""
    config = {
        "benchmarks": {
            "retrieval": {
                "enabled": True,
                "methods": ["bm25", "semantic", "hybrid"],
                "dataset": "sample_test_dataset.json"
            },
            "rag": {
                "enabled": True,
                "systems": [
                    ["BM25 + GPT-3.5", "bm25", "gpt-3.5"],
                    ["Hybrid + GPT-4", "hybrid", "gpt-4"]
                ],
                "dataset": "rag_test_dataset.json"
            },
            "agents": {
                "enabled": True,
                "agents": [
                    "Dev Copilot Agent",
                    "Enterprise Assistant Agent",
                    "Analytics Agent"
                ],
                "tasks": "agent_benchmark_tasks.json"
            }
        },
        "output": {
            "generate_html_reports": True,
            "generate_json_summary": True,
            "generate_csv_export": True
        }
    }
    
    with open("benchmark_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Default configuration created: benchmark_config.json")

def main():
    """Función principal del orchestrator"""
    print("🚀 Portal 4 - Benchmark Suite Orchestrator")
    print("="*60)
    
    # Crear configuración por defecto si no existe
    if not Path("benchmark_config.json").exists():
        create_default_config()
    
    # Ejecutar suite de benchmarks
    suite = Portal4BenchmarkSuite("benchmark_config.json")
    results = suite.run_full_benchmark_suite()
    
    print("\n🎉 Benchmark Suite Execution Complete!")
    print(f"📊 Results directory: {suite.results_dir}")
    print("\nNext steps:")
    print("1. Review the consolidated report")
    print("2. Analyze individual benchmark reports") 
    print("3. Use results to optimize your AI systems")

if __name__ == "__main__":
    main()
