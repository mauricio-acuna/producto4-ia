#!/usr/bin/env python3
"""
Agent Benchmark Suite para Portal 4
Evalúa sistemas de agentes AI en tareas específicas
"""

import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from enum import Enum

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    DOCUMENT_ANALYSIS = "document_analysis"
    PROBLEM_SOLVING = "problem_solving"
    DECISION_MAKING = "decision_making"

@dataclass
class AgentTask:
    """Tarea individual para evaluar un agente"""
    task_id: str
    task_type: TaskType
    description: str
    input_data: Dict[str, Any]
    expected_output: Any
    evaluation_criteria: List[str]
    max_time_limit: float = 30.0  # seconds

@dataclass
class AgentResult:
    """Resultado de ejecución de una tarea por un agente"""
    task_id: str
    agent_name: str
    output: Any
    execution_time: float
    success: bool
    accuracy_score: float
    efficiency_score: float
    error_message: Optional[str] = None

@dataclass
class AgentBenchmarkResult:
    """Resultado completo del benchmark para un agente"""
    agent_name: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    avg_accuracy: float
    avg_efficiency: float
    avg_execution_time: float
    task_type_performance: Dict[str, float]

class AgentBenchmark:
    """Suite de benchmark para agentes AI"""
    
    def __init__(self, tasks_file: str):
        self.tasks_file = Path(tasks_file)
        self.tasks = self._load_tasks()
        self.agents = {}
        
    def _load_tasks(self) -> List[AgentTask]:
        """Carga tareas de benchmark desde archivo"""
        with open(self.tasks_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tasks = []
        for task_data in data:
            task = AgentTask(
                task_id=task_data["id"],
                task_type=TaskType(task_data["type"]),
                description=task_data["description"],
                input_data=task_data["input"],
                expected_output=task_data["expected_output"],
                evaluation_criteria=task_data["evaluation_criteria"],
                max_time_limit=task_data.get("max_time_limit", 30.0)
            )
            tasks.append(task)
        
        print(f"Loaded {len(tasks)} benchmark tasks")
        return tasks
    
    def register_agent(self, name: str, agent_function):
        """Registra un agente para evaluación"""
        self.agents[name] = agent_function
        print(f"Registered agent: {name}")
    
    def _simulate_agent_execution(self, agent_name: str, task: AgentTask) -> AgentResult:
        """Simula la ejecución de un agente en una tarea"""
        start_time = time.time()
        
        try:
            # Simulación de diferentes tipos de agentes
            if "copilot" in agent_name.lower():
                output = self._simulate_copilot_agent(task)
            elif "assistant" in agent_name.lower():
                output = self._simulate_assistant_agent(task)
            elif "analyst" in agent_name.lower():
                output = self._simulate_analyst_agent(task)
            else:
                output = self._simulate_generic_agent(task)
            
            execution_time = time.time() - start_time
            
            # Verificar timeout
            if execution_time > task.max_time_limit:
                return AgentResult(
                    task_id=task.task_id,
                    agent_name=agent_name,
                    output=None,
                    execution_time=execution_time,
                    success=False,
                    accuracy_score=0.0,
                    efficiency_score=0.0,
                    error_message="Timeout exceeded"
                )
            
            # Evaluar resultado
            accuracy = self._evaluate_accuracy(task, output)
            efficiency = self._evaluate_efficiency(execution_time, task.max_time_limit)
            
            return AgentResult(
                task_id=task.task_id,
                agent_name=agent_name,
                output=output,
                execution_time=execution_time,
                success=True,
                accuracy_score=accuracy,
                efficiency_score=efficiency
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            return AgentResult(
                task_id=task.task_id,
                agent_name=agent_name,
                output=None,
                execution_time=execution_time,
                success=False,
                accuracy_score=0.0,
                efficiency_score=0.0,
                error_message=str(e)
            )
    
    def _simulate_copilot_agent(self, task: AgentTask) -> Any:
        """Simula agente copiloto especializado en código"""
        if task.task_type == TaskType.CODE_GENERATION:
            # Simular generación de código de alta calidad
            return {
                "code": f"def solution():\n    # High-quality code for: {task.description}\n    return result",
                "tests": ["test_case_1", "test_case_2"],
                "documentation": f"This function solves: {task.description}"
            }
        else:
            # Adaptación básica a otras tareas
            return f"Code-oriented solution for: {task.description}"
    
    def _simulate_assistant_agent(self, task: AgentTask) -> Any:
        """Simula agente asistente empresarial"""
        if task.task_type == TaskType.DOCUMENT_ANALYSIS:
            return {
                "summary": f"Analysis of document related to: {task.description}",
                "key_points": ["Point 1", "Point 2", "Point 3"],
                "recommendations": ["Action 1", "Action 2"]
            }
        elif task.task_type == TaskType.DECISION_MAKING:
            return {
                "decision": f"Recommended decision for: {task.description}",
                "reasoning": "Based on business analysis and risk assessment",
                "alternatives": ["Option A", "Option B"]
            }
        else:
            return f"Business-oriented solution for: {task.description}"
    
    def _simulate_analyst_agent(self, task: AgentTask) -> Any:
        """Simula agente analista de datos"""
        if task.task_type == TaskType.PROBLEM_SOLVING:
            return {
                "analysis": f"Statistical analysis for: {task.description}",
                "insights": ["Insight 1", "Insight 2"],
                "metrics": {"accuracy": 0.85, "confidence": 0.92}
            }
        else:
            return f"Data-driven solution for: {task.description}"
    
    def _simulate_generic_agent(self, task: AgentTask) -> Any:
        """Simula agente genérico"""
        return f"Generic solution for: {task.description}"
    
    def _evaluate_accuracy(self, task: AgentTask, output: Any) -> float:
        """Evalúa precisión del resultado"""
        # Simulación de evaluación de precisión
        if output is None:
            return 0.0
        
        # Factores que afectan la precisión
        accuracy_factors = []
        
        # Verificar si es el tipo de output esperado
        expected_type = type(task.expected_output)
        actual_type = type(output)
        
        if expected_type == actual_type:
            accuracy_factors.append(0.3)
        
        # Verificar contenido relevante (simulado)
        if isinstance(output, dict) and isinstance(task.expected_output, dict):
            # Comparar keys
            expected_keys = set(task.expected_output.keys())
            actual_keys = set(output.keys())
            key_overlap = len(expected_keys.intersection(actual_keys)) / len(expected_keys)
            accuracy_factors.append(key_overlap * 0.4)
        
        elif isinstance(output, str) and isinstance(task.expected_output, str):
            # Comparar contenido textual básico
            expected_words = set(task.expected_output.lower().split())
            actual_words = set(str(output).lower().split())
            word_overlap = len(expected_words.intersection(actual_words)) / max(len(expected_words), 1)
            accuracy_factors.append(word_overlap * 0.4)
        
        # Factor de completitud
        if str(output).strip():  # No vacío
            accuracy_factors.append(0.3)
        
        return min(1.0, sum(accuracy_factors))
    
    def _evaluate_efficiency(self, execution_time: float, max_time: float) -> float:
        """Evalúa eficiencia basada en tiempo de ejecución"""
        if execution_time >= max_time:
            return 0.0
        
        # Score de eficiencia inverso al tiempo (más rápido = mejor)
        efficiency = 1.0 - (execution_time / max_time)
        return max(0.0, efficiency)
    
    def evaluate_agent(self, agent_name: str) -> AgentBenchmarkResult:
        """Evalúa un agente específico en todas las tareas"""
        print(f"Evaluating agent: {agent_name}")
        
        results = []
        task_type_scores = {}
        
        for task in self.tasks:
            result = self._simulate_agent_execution(agent_name, task)
            results.append(result)
            
            # Agrupar por tipo de tarea
            task_type = task.task_type.value
            if task_type not in task_type_scores:
                task_type_scores[task_type] = []
            
            if result.success:
                task_type_scores[task_type].append(result.accuracy_score)
        
        # Calcular métricas agregadas
        successful_results = [r for r in results if r.success]
        
        total_tasks = len(results)
        successful_tasks = len(successful_results)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
        
        avg_accuracy = sum(r.accuracy_score for r in successful_results) / len(successful_results) if successful_results else 0.0
        avg_efficiency = sum(r.efficiency_score for r in successful_results) / len(successful_results) if successful_results else 0.0
        avg_execution_time = sum(r.execution_time for r in results) / len(results) if results else 0.0
        
        # Promedio por tipo de tarea
        task_type_performance = {}
        for task_type, scores in task_type_scores.items():
            task_type_performance[task_type] = sum(scores) / len(scores) if scores else 0.0
        
        return AgentBenchmarkResult(
            agent_name=agent_name,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            success_rate=success_rate,
            avg_accuracy=avg_accuracy,
            avg_efficiency=avg_efficiency,
            avg_execution_time=avg_execution_time,
            task_type_performance=task_type_performance
        )
    
    def run_comparative_benchmark(self, agent_names: List[str]) -> Dict[str, AgentBenchmarkResult]:
        """Ejecuta benchmark comparativo de múltiples agentes"""
        print("Starting comparative agent benchmark...")
        
        results = {}
        
        for agent_name in agent_names:
            result = self.evaluate_agent(agent_name)
            results[agent_name] = result
            print(f"✅ Completed evaluation: {agent_name}")
        
        return results
    
    def generate_report(self, results: Dict[str, AgentBenchmarkResult], 
                       output_path: Optional[str] = None) -> str:
        """Genera reporte de benchmark de agentes"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = f"agent_benchmark_report_{timestamp}.html"
        
        # Encontrar mejores agentes por categoría
        best_overall = max(results.items(), key=lambda x: x[1].avg_accuracy)
        best_efficiency = max(results.items(), key=lambda x: x[1].avg_efficiency)
        most_reliable = max(results.items(), key=lambda x: x[1].success_rate)
        
        # Crear tabla de resultados
        table_rows = ""
        for agent_name, result in results.items():
            table_rows += f"""
            <tr>
                <td>{agent_name}</td>
                <td>{result.success_rate:.3f}</td>
                <td>{result.avg_accuracy:.3f}</td>
                <td>{result.avg_efficiency:.3f}</td>
                <td>{result.avg_execution_time:.3f}s</td>
                <td>{result.successful_tasks}/{result.total_tasks}</td>
            </tr>
            """
        
        # Crear sección de performance por tipo de tarea
        task_performance = ""
        all_task_types = set()
        for result in results.values():
            all_task_types.update(result.task_type_performance.keys())
        
        for task_type in sorted(all_task_types):
            task_performance += f"<h4>{task_type.replace('_', ' ').title()}</h4><ul>"
            for agent_name, result in results.items():
                score = result.task_type_performance.get(task_type, 0.0)
                task_performance += f"<li>{agent_name}: {score:.3f}</li>"
            task_performance += "</ul>"
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Agent Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                .winner {{ background-color: #d4edda; }}
                .section {{ margin: 30px 0; }}
                .metric-box {{ background: #f8f9fa; padding: 20px; margin: 10px 0; border-radius: 5px; }}
                .highlight {{ color: #28a745; font-weight: bold; }}
            </style>
        </head>
        <body>
            <h1>🤖 Agent Benchmark Report</h1>
            
            <div class="section">
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Tasks File:</strong> {self.tasks_file}</p>
                <p><strong>Total Benchmark Tasks:</strong> {len(self.tasks)}</p>
                <p><strong>Evaluated Agents:</strong> {len(results)}</p>
            </div>
            
            <h2>📊 Overall Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th>Success Rate</th>
                        <th>Avg Accuracy</th>
                        <th>Avg Efficiency</th>
                        <th>Avg Execution Time</th>
                        <th>Tasks Completed</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            
            <h2>🏆 Best Performers</h2>
            <div class="metric-box">
                <h3>Best Overall Accuracy</h3>
                <p class="highlight">{best_overall[0]}</p>
                <p>Average Accuracy: {best_overall[1].avg_accuracy:.3f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Most Efficient</h3>
                <p class="highlight">{best_efficiency[0]}</p>
                <p>Average Efficiency: {best_efficiency[1].avg_efficiency:.3f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Most Reliable</h3>
                <p class="highlight">{most_reliable[0]}</p>
                <p>Success Rate: {most_reliable[1].success_rate:.3f}</p>
            </div>
            
            <h2>📋 Performance by Task Type</h2>
            <div class="section">
                {task_performance}
            </div>
            
            <h2>💡 Recommendations</h2>
            <div class="section">
                <ul>
                    <li><strong>For Critical Tasks:</strong> Use {most_reliable[0]} (highest reliability)</li>
                    <li><strong>For Quality Results:</strong> Use {best_overall[0]} (highest accuracy)</li>
                    <li><strong>For Speed Requirements:</strong> Use {best_efficiency[0]} (most efficient)</li>
                </ul>
            </div>
            
            <h2>📈 Methodology</h2>
            <div class="section">
                <p><strong>Success Rate:</strong> Percentage of tasks completed without errors</p>
                <p><strong>Accuracy:</strong> Quality of output compared to expected results</p>
                <p><strong>Efficiency:</strong> Speed of execution relative to time limits</p>
                <p><strong>Task Types:</strong> {', '.join(sorted(all_task_types))}</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"Agent benchmark report generated: {output_path}")
        return output_path
    
    def print_summary(self, results: Dict[str, AgentBenchmarkResult]):
        """Imprime resumen de resultados"""
        print("\n" + "="*70)
        print("AGENT BENCHMARK RESULTS")
        print("="*70)
        
        for agent_name, result in results.items():
            print(f"\n🤖 {agent_name}:")
            print(f"  ✅ Success Rate: {result.success_rate:.3f}")
            print(f"  🎯 Avg Accuracy: {result.avg_accuracy:.3f}")
            print(f"  ⚡ Avg Efficiency: {result.avg_efficiency:.3f}")
            print(f"  ⏱️  Avg Time: {result.avg_execution_time:.3f}s")
            print(f"  📊 Completed: {result.successful_tasks}/{result.total_tasks}")
        
        # Mejor agente overall
        best_agent = max(results.items(), key=lambda x: x[1].avg_accuracy)
        print(f"\n🏆 BEST OVERALL: {best_agent[0]} (Accuracy: {best_agent[1].avg_accuracy:.3f})")

def create_agent_benchmark_tasks():
    """Crea dataset de tareas para benchmark de agentes"""
    tasks = [
        {
            "id": "task_001",
            "type": "code_generation",
            "description": "Generate a Python function to calculate Fibonacci sequence",
            "input": {"n": 10, "language": "python"},
            "expected_output": {"code": "def fibonacci(n): ...", "tests": [], "documentation": ""},
            "evaluation_criteria": ["correctness", "efficiency", "readability"],
            "max_time_limit": 15.0
        },
        {
            "id": "task_002",
            "type": "document_analysis",
            "description": "Analyze a business proposal and extract key insights",
            "input": {"document": "Business proposal for new product launch..."},
            "expected_output": {"summary": "", "key_points": [], "recommendations": []},
            "evaluation_criteria": ["completeness", "accuracy", "actionability"],
            "max_time_limit": 25.0
        },
        {
            "id": "task_003",
            "type": "problem_solving",
            "description": "Optimize inventory management for e-commerce platform",
            "input": {"current_inventory": 1000, "demand_forecast": [100, 150, 200]},
            "expected_output": {"strategy": "", "metrics": {}, "timeline": []},
            "evaluation_criteria": ["feasibility", "impact", "implementation"],
            "max_time_limit": 30.0
        },
        {
            "id": "task_004",
            "type": "decision_making",
            "description": "Choose optimal cloud architecture for new application",
            "input": {"requirements": ["scalability", "cost-efficiency", "security"]},
            "expected_output": {"decision": "", "reasoning": "", "alternatives": []},
            "evaluation_criteria": ["justification", "completeness", "practicality"],
            "max_time_limit": 20.0
        },
        {
            "id": "task_005",
            "type": "code_generation",
            "description": "Create a REST API endpoint for user authentication",
            "input": {"framework": "FastAPI", "security": "JWT"},
            "expected_output": {"code": "", "tests": [], "documentation": ""},
            "evaluation_criteria": ["security", "functionality", "best_practices"],
            "max_time_limit": 25.0
        }
    ] * 4  # Repeat for larger dataset
    
    # Update IDs to be unique
    for i, task in enumerate(tasks):
        task["id"] = f"task_{i+1:03d}"
    
    with open("agent_benchmark_tasks.json", "w", encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    
    print("Agent benchmark tasks created: agent_benchmark_tasks.json")

def main():
    """Función principal para ejecutar benchmark de agentes"""
    # Crear tasks si no existen
    if not Path("agent_benchmark_tasks.json").exists():
        create_agent_benchmark_tasks()
    
    # Inicializar benchmark
    benchmark = AgentBenchmark("agent_benchmark_tasks.json")
    
    # Definir agentes a evaluar (simulados)
    agent_names = [
        "Dev Copilot Agent",
        "Enterprise Assistant Agent", 
        "Legal Finance Agent",
        "Analytics Agent",
        "Generic AI Agent"
    ]
    
    # Ejecutar benchmark comparativo
    results = benchmark.run_comparative_benchmark(agent_names)
    
    # Mostrar resultados
    benchmark.print_summary(results)
    
    # Generar reporte
    report_path = benchmark.generate_report(results)
    print(f"\n📊 Agent benchmark report: {report_path}")

if __name__ == "__main__":
    main()
