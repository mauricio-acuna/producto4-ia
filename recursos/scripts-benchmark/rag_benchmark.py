#!/usr/bin/env python3
"""
RAG Benchmark Suite para Portal 4
Evalúa sistemas RAG completos: Retrieval + Generation
"""

import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from collections import defaultdict

@dataclass
class RAGResult:
    """Resultado de una consulta RAG completa"""
    query_id: str
    query_text: str
    retrieved_contexts: List[Dict[str, Any]]
    generated_answer: str
    ground_truth_answer: str
    response_time: float
    retrieval_time: float
    generation_time: float
    context_relevance_scores: List[float]

@dataclass
class RAGBenchmarkResult:
    """Resultado completo del benchmark RAG"""
    system_name: str
    
    # Métricas de Retrieval
    retrieval_precision_at_3: float
    retrieval_precision_at_5: float
    context_relevance: float
    
    # Métricas de Generation
    answer_relevance: float
    faithfulness: float
    answer_similarity: float
    
    # Métricas de Performance
    avg_total_time: float
    avg_retrieval_time: float
    avg_generation_time: float
    
    # Estadísticas
    total_queries: int
    successful_queries: int

class RAGBenchmark:
    """Suite de benchmark para sistemas RAG"""
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.test_data = self._load_test_dataset()
        self.results = []
        
    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Carga dataset con queries, contexts relevantes y respuestas esperadas"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} RAG test cases from {self.dataset_path}")
        return data
    
    def _simulate_retrieval(self, query: str, method: str = "hybrid") -> Tuple[List[Dict], float]:
        """Simula retrieval de contextos relevantes"""
        start_time = time.time()
        
        # Simulación de diferentes estrategias de retrieval
        import random
        seed_offset = {"bm25": 0, "semantic": 100, "hybrid": 200}.get(method, 0)
        random.seed(hash(query) % 1000 + seed_offset)
        
        contexts = []
        for i in range(5):  # Top 5 contexts
            relevance_score = random.uniform(0.3, 0.9) if method == "hybrid" else random.uniform(0.2, 0.8)
            contexts.append({
                "id": f"ctx_{i}_{method}",
                "content": f"Sample context {i} for query: {query[:30]}...",
                "source": f"document_{i}.pdf",
                "relevance_score": relevance_score,
                "chunk_index": i
            })
        
        # Ordenar por relevancia
        contexts = sorted(contexts, key=lambda x: x["relevance_score"], reverse=True)
        
        retrieval_time = time.time() - start_time
        return contexts, retrieval_time
    
    def _simulate_generation(self, query: str, contexts: List[Dict], model: str = "gpt-3.5") -> Tuple[str, float]:
        """Simula generación de respuesta basada en contextos"""
        start_time = time.time()
        
        # Simular diferentes calidades de respuesta según el modelo
        quality_multiplier = {
            "gpt-4": 0.95,
            "gpt-3.5": 0.85,
            "llama2": 0.75,
            "local-model": 0.65
        }.get(model, 0.7)
        
        # Simular tiempo de generación basado en modelo
        base_time = {
            "gpt-4": 0.8,
            "gpt-3.5": 0.5,
            "llama2": 1.2,
            "local-model": 2.0
        }.get(model, 1.0)
        
        # Simular generación (en real: llamada a LLM)
        time.sleep(base_time * 0.1)  # Simulated generation delay
        
        # Crear respuesta simulada basada en query y contextos
        context_quality = sum(ctx["relevance_score"] for ctx in contexts[:3]) / 3
        response_quality = context_quality * quality_multiplier
        
        if response_quality > 0.8:
            answer = f"Based on the provided context, {query.lower().replace('?', '')} can be addressed as follows: This is a comprehensive answer that properly utilizes the retrieved context to provide accurate information."
        elif response_quality > 0.6:
            answer = f"According to the available information, {query.lower().replace('?', '')} has the following details: This is a reasonable answer though it may miss some nuances."
        else:
            answer = f"Regarding {query.lower().replace('?', '')}, the information suggests: This is a basic answer that may lack depth or accuracy."
        
        generation_time = time.time() - start_time
        return answer, generation_time
    
    def _calculate_context_relevance(self, query: str, contexts: List[Dict], 
                                   ground_truth_contexts: List[str]) -> List[float]:
        """Calcula relevancia de contextos recuperados"""
        relevance_scores = []
        
        for context in contexts:
            # En implementación real: usar modelo de embedding o LLM para evaluar relevancia
            # Simulación: basada en overlap de keywords y ground truth
            
            # Verificar si el contexto está en ground truth
            is_relevant = any(gt in context.get("content", "") or 
                            context.get("id", "") in gt 
                            for gt in ground_truth_contexts)
            
            if is_relevant:
                base_score = 0.8
            else:
                # Simular relevancia basada en keywords
                query_words = set(query.lower().split())
                context_words = set(context.get("content", "").lower().split())
                overlap = len(query_words.intersection(context_words))
                base_score = min(0.7, overlap * 0.1)
            
            relevance_scores.append(base_score)
        
        return relevance_scores
    
    def _calculate_answer_relevance(self, query: str, answer: str, ground_truth: str) -> float:
        """Evalúa qué tan relevante es la respuesta para la pregunta"""
        # Simulación de answer relevance
        # En implementación real: usar embeddings o LLM-as-judge
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        gt_words = set(ground_truth.lower().split())
        
        # Verificar overlap con query
        query_overlap = len(query_words.intersection(answer_words)) / len(query_words)
        
        # Verificar similaridad con ground truth
        gt_overlap = len(gt_words.intersection(answer_words)) / len(gt_words)
        
        # Combinar métricas
        relevance_score = (query_overlap * 0.3 + gt_overlap * 0.7)
        return min(1.0, relevance_score)
    
    def _calculate_faithfulness(self, answer: str, contexts: List[Dict]) -> float:
        """Evalúa si la respuesta es fiel a los contextos proporcionados"""
        # Simulación de faithfulness
        # En implementación real: verificar que claims en answer estén en contexts
        
        answer_words = set(answer.lower().split())
        context_words = set()
        
        for context in contexts:
            context_words.update(context.get("content", "").lower().split())
        
        # Verificar overlap de información factual (simulado)
        factual_overlap = len(answer_words.intersection(context_words)) / len(answer_words)
        
        # Penalizar respuestas que parecen inventar información
        fabrication_penalty = 0.0
        if "according to" in answer.lower() and factual_overlap < 0.3:
            fabrication_penalty = 0.2
        
        faithfulness_score = max(0.0, factual_overlap - fabrication_penalty)
        return min(1.0, faithfulness_score)
    
    def _calculate_answer_similarity(self, generated_answer: str, ground_truth: str) -> float:
        """Calcula similaridad semántica entre respuesta generada y ground truth"""
        # Simulación de similaridad semántica
        # En implementación real: usar embeddings o métricas como BLEU, ROUGE
        
        # Normalizar textos
        gen_clean = re.sub(r'[^\w\s]', '', generated_answer.lower())
        gt_clean = re.sub(r'[^\w\s]', '', ground_truth.lower())
        
        gen_words = set(gen_clean.split())
        gt_words = set(gt_clean.split())
        
        if not gt_words:
            return 0.0
        
        # Jaccard similarity
        intersection = len(gen_words.intersection(gt_words))
        union = len(gen_words.union(gt_words))
        
        similarity = intersection / union if union > 0 else 0.0
        return similarity
    
    def evaluate_rag_system(self, system_name: str, 
                          retrieval_method: str = "hybrid",
                          generation_model: str = "gpt-3.5") -> RAGBenchmarkResult:
        """Evalúa un sistema RAG completo"""
        print(f"Evaluating RAG system: {system_name}")
        print(f"  Retrieval: {retrieval_method}")
        print(f"  Generation: {generation_model}")
        
        results = []
        
        for test_case in self.test_data:
            query = test_case["query"]
            ground_truth_answer = test_case["expected_answer"]
            ground_truth_contexts = test_case.get("relevant_contexts", [])
            
            # Paso 1: Retrieval
            contexts, retrieval_time = self._simulate_retrieval(query, retrieval_method)
            
            # Paso 2: Generation
            generated_answer, generation_time = self._simulate_generation(
                query, contexts, generation_model
            )
            
            # Calcular métricas
            context_relevance_scores = self._calculate_context_relevance(
                query, contexts, ground_truth_contexts
            )
            
            total_time = retrieval_time + generation_time
            
            result = RAGResult(
                query_id=test_case["id"],
                query_text=query,
                retrieved_contexts=contexts,
                generated_answer=generated_answer,
                ground_truth_answer=ground_truth_answer,
                response_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                context_relevance_scores=context_relevance_scores
            )
            
            results.append(result)
        
        # Calcular métricas agregadas
        return self._aggregate_results(results, system_name)
    
    def _aggregate_results(self, results: List[RAGResult], system_name: str) -> RAGBenchmarkResult:
        """Agrega resultados individuales en métricas del sistema"""
        
        # Métricas de Retrieval
        precision_3_scores = []
        precision_5_scores = []
        context_relevance_scores = []
        
        # Métricas de Generation
        answer_relevance_scores = []
        faithfulness_scores = []
        similarity_scores = []
        
        # Métricas de Performance
        total_times = []
        retrieval_times = []
        generation_times = []
        
        successful_queries = 0
        
        for result in results:
            try:
                # Retrieval metrics
                top_3_relevance = result.context_relevance_scores[:3]
                top_5_relevance = result.context_relevance_scores[:5]
                
                precision_3_scores.append(sum(1 for score in top_3_relevance if score > 0.5) / min(3, len(top_3_relevance)))
                precision_5_scores.append(sum(1 for score in top_5_relevance if score > 0.5) / min(5, len(top_5_relevance)))
                context_relevance_scores.extend(result.context_relevance_scores)
                
                # Generation metrics
                answer_rel = self._calculate_answer_relevance(
                    result.query_text, result.generated_answer, result.ground_truth_answer
                )
                faith = self._calculate_faithfulness(result.generated_answer, result.retrieved_contexts)
                similarity = self._calculate_answer_similarity(
                    result.generated_answer, result.ground_truth_answer
                )
                
                answer_relevance_scores.append(answer_rel)
                faithfulness_scores.append(faith)
                similarity_scores.append(similarity)
                
                # Performance metrics
                total_times.append(result.response_time)
                retrieval_times.append(result.retrieval_time)
                generation_times.append(result.generation_time)
                
                successful_queries += 1
                
            except Exception as e:
                print(f"Error processing query {result.query_id}: {e}")
                continue
        
        # Calcular promedios
        def safe_mean(scores):
            return sum(scores) / len(scores) if scores else 0.0
        
        return RAGBenchmarkResult(
            system_name=system_name,
            retrieval_precision_at_3=safe_mean(precision_3_scores),
            retrieval_precision_at_5=safe_mean(precision_5_scores),
            context_relevance=safe_mean(context_relevance_scores),
            answer_relevance=safe_mean(answer_relevance_scores),
            faithfulness=safe_mean(faithfulness_scores),
            answer_similarity=safe_mean(similarity_scores),
            avg_total_time=safe_mean(total_times),
            avg_retrieval_time=safe_mean(retrieval_times),
            avg_generation_time=safe_mean(generation_times),
            total_queries=len(results),
            successful_queries=successful_queries
        )
    
    def run_comparative_benchmark(self) -> Dict[str, RAGBenchmarkResult]:
        """Ejecuta benchmark comparativo de múltiples configuraciones"""
        print("Starting comparative RAG benchmark...")
        
        configurations = [
            ("BM25 + GPT-3.5", "bm25", "gpt-3.5"),
            ("Semantic + GPT-3.5", "semantic", "gpt-3.5"),
            ("Hybrid + GPT-3.5", "hybrid", "gpt-3.5"),
            ("Hybrid + GPT-4", "hybrid", "gpt-4"),
            ("Hybrid + Local Model", "hybrid", "local-model")
        ]
        
        results = {}
        
        for system_name, retrieval_method, generation_model in configurations:
            result = self.evaluate_rag_system(system_name, retrieval_method, generation_model)
            results[system_name] = result
            print(f"✅ Completed: {system_name}")
        
        return results
    
    def generate_detailed_report(self, results: Dict[str, RAGBenchmarkResult], 
                               output_path: Optional[str] = None) -> str:
        """Genera reporte detallado de benchmark RAG"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = f"rag_benchmark_report_{timestamp}.html"
        
        # Encontrar mejor sistema por cada métrica
        best_systems = {
            "retrieval": max(results.items(), key=lambda x: x[1].retrieval_precision_at_5),
            "generation": max(results.items(), key=lambda x: x[1].answer_relevance),
            "faithfulness": max(results.items(), key=lambda x: x[1].faithfulness),
            "speed": min(results.items(), key=lambda x: x[1].avg_total_time),
            "overall": max(results.items(), key=lambda x: (x[1].answer_relevance + x[1].faithfulness) / 2)
        }
        
        # Crear tabla de resultados
        table_rows = ""
        for system_name, result in results.items():
            table_rows += f"""
            <tr>
                <td>{system_name}</td>
                <td>{result.retrieval_precision_at_5:.3f}</td>
                <td>{result.context_relevance:.3f}</td>
                <td>{result.answer_relevance:.3f}</td>
                <td>{result.faithfulness:.3f}</td>
                <td>{result.answer_similarity:.3f}</td>
                <td>{result.avg_total_time:.3f}s</td>
                <td>{result.successful_queries}/{result.total_queries}</td>
            </tr>
            """
        
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Benchmark Report</title>
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
            <h1>🤖 RAG Systems Benchmark Report</h1>
            
            <div class="section">
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Dataset:</strong> {self.dataset_path}</p>
                <p><strong>Total Test Cases:</strong> {len(self.test_data)}</p>
            </div>
            
            <h2>📊 Results Summary</h2>
            <table>
                <thead>
                    <tr>
                        <th>System</th>
                        <th>Retrieval P@5</th>
                        <th>Context Relevance</th>
                        <th>Answer Relevance</th>
                        <th>Faithfulness</th>
                        <th>Answer Similarity</th>
                        <th>Avg Response Time</th>
                        <th>Success Rate</th>
                    </tr>
                </thead>
                <tbody>
                    {table_rows}
                </tbody>
            </table>
            
            <h2>🏆 Best Performers</h2>
            <div class="metric-box">
                <h3>Best Retrieval Performance</h3>
                <p class="highlight">{best_systems["retrieval"][0]}</p>
                <p>Precision@5: {best_systems["retrieval"][1].retrieval_precision_at_5:.3f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Best Generation Quality</h3>
                <p class="highlight">{best_systems["generation"][0]}</p>
                <p>Answer Relevance: {best_systems["generation"][1].answer_relevance:.3f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Most Faithful System</h3>
                <p class="highlight">{best_systems["faithfulness"][0]}</p>
                <p>Faithfulness: {best_systems["faithfulness"][1].faithfulness:.3f}</p>
            </div>
            
            <div class="metric-box">
                <h3>Fastest System</h3>
                <p class="highlight">{best_systems["speed"][0]}</p>
                <p>Avg Response Time: {best_systems["speed"][1].avg_total_time:.3f}s</p>
            </div>
            
            <h2>💡 Recommendations</h2>
            <div class="section">
                <h3>For Production Use:</h3>
                <ul>
                    <li><strong>Best Overall:</strong> {best_systems["overall"][0]} - Balanced performance across quality metrics</li>
                    <li><strong>High Accuracy Needed:</strong> {best_systems["faithfulness"][0]} - Highest faithfulness to source material</li>
                    <li><strong>Low Latency Required:</strong> {best_systems["speed"][0]} - Fastest response times</li>
                </ul>
                
                <h3>Optimization Opportunities:</h3>
                <ul>
                    <li>Consider hybrid retrieval approaches for better context selection</li>
                    <li>Implement reranking for improved context relevance</li>
                    <li>Fine-tune generation models for domain-specific accuracy</li>
                    <li>Optimize chunk size and overlap for better retrieval performance</li>
                </ul>
            </div>
            
            <h2>📈 Metric Definitions</h2>
            <div class="section">
                <p><strong>Retrieval P@5:</strong> Precision at 5 - fraction of top 5 retrieved contexts that are relevant</p>
                <p><strong>Context Relevance:</strong> Average relevance score of retrieved contexts to the query</p>
                <p><strong>Answer Relevance:</strong> How well the generated answer addresses the original question</p>
                <p><strong>Faithfulness:</strong> Degree to which the answer is grounded in the provided contexts</p>
                <p><strong>Answer Similarity:</strong> Semantic similarity between generated and expected answers</p>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"RAG benchmark report generated: {output_path}")
        return output_path
    
    def print_summary(self, results: Dict[str, RAGBenchmarkResult]):
        """Imprime resumen de resultados en consola"""
        print("\n" + "="*80)
        print("RAG BENCHMARK RESULTS")
        print("="*80)
        
        for system_name, result in results.items():
            print(f"\n{system_name}:")
            print(f"  📥 Retrieval P@5: {result.retrieval_precision_at_5:.3f}")
            print(f"  🎯 Context Relevance: {result.context_relevance:.3f}")
            print(f"  💬 Answer Relevance: {result.answer_relevance:.3f}")
            print(f"  ✅ Faithfulness: {result.faithfulness:.3f}")
            print(f"  📝 Answer Similarity: {result.answer_similarity:.3f}")
            print(f"  ⚡ Avg Response Time: {result.avg_total_time:.3f}s")
            print(f"  ✔️  Success Rate: {result.successful_queries}/{result.total_queries}")
        
        # Encontrar mejor sistema overall
        best_overall = max(results.items(), 
                          key=lambda x: (x[1].answer_relevance + x[1].faithfulness) / 2)
        
        print(f"\n🏆 OVERALL WINNER: {best_overall[0]}")
        print(f"   Combined Score: {(best_overall[1].answer_relevance + best_overall[1].faithfulness) / 2:.3f}")

def create_rag_test_dataset():
    """Crea dataset de prueba para RAG benchmark"""
    rag_test_data = [
        {
            "id": "rag_001",
            "query": "How many vacation days do new employees get?",
            "expected_answer": "New employees receive 15 vacation days in their first year, increasing to 20 days after 2 years of employment.",
            "relevant_contexts": [
                "New employee benefits include 15 vacation days",
                "Vacation policy for first year employees",
                "After 2 years, vacation days increase to 20"
            ]
        },
        {
            "id": "rag_002",
            "query": "What is the remote work policy for developers?",
            "expected_answer": "Developers can work remotely up to 3 days per week with manager approval. Full remote work requires special circumstances and executive approval.",
            "relevant_contexts": [
                "Developer remote work guidelines",
                "3 days per week remote work policy",
                "Manager approval required for remote work"
            ]
        },
        {
            "id": "rag_003",
            "query": "What are the security requirements for handling customer data?",
            "expected_answer": "Customer data must be encrypted at rest and in transit, access requires multi-factor authentication, and all data access is logged and audited monthly.",
            "relevant_contexts": [
                "Customer data encryption requirements",
                "Multi-factor authentication for data access",
                "Monthly audit logs for customer data"
            ]
        },
        {
            "id": "rag_004",
            "query": "How do I submit a bug report for the internal tools?",
            "expected_answer": "Bug reports should be submitted through the internal JIRA system with steps to reproduce, expected vs actual behavior, and environment details.",
            "relevant_contexts": [
                "Internal bug reporting process",
                "JIRA system for bug tracking",
                "Required information for bug reports"
            ]
        },
        {
            "id": "rag_005",
            "query": "What is the process for requesting new software licenses?",
            "expected_answer": "Software license requests must be submitted through ServiceNow with business justification, cost approval from manager, and security review for new vendors.",
            "relevant_contexts": [
                "Software license request process",
                "ServiceNow license management",
                "Manager approval for software costs"
            ]
        }
    ] * 6  # Repeat for larger dataset
    
    # Update IDs to be unique
    for i, item in enumerate(rag_test_data):
        item["id"] = f"rag_{i+1:03d}"
    
    with open("rag_test_dataset.json", "w", encoding='utf-8') as f:
        json.dump(rag_test_data, f, indent=2, ensure_ascii=False)
    
    print("RAG test dataset created: rag_test_dataset.json")

def main():
    """Función principal para ejecutar RAG benchmark"""
    # Crear dataset de prueba si no existe
    if not Path("rag_test_dataset.json").exists():
        create_rag_test_dataset()
    
    # Configuración del benchmark
    config = {
        "retrieval": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5
        },
        "generation": {
            "max_tokens": 256,
            "temperature": 0.1
        }
    }
    
    # Ejecutar benchmark
    benchmark = RAGBenchmark("rag_test_dataset.json", config)
    results = benchmark.run_comparative_benchmark()
    
    # Mostrar resultados
    benchmark.print_summary(results)
    
    # Generar reporte detallado
    report_path = benchmark.generate_detailed_report(results)
    print(f"\n📊 Detailed RAG benchmark report: {report_path}")

if __name__ == "__main__":
    main()
