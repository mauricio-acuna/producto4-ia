#!/usr/bin/env python3
"""
Retrieval Benchmark Suite para Portal 4
Evalúa y compara diferentes estrategias de retrieval: BM25, Semantic, Hybrid
"""

import json
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Simulated imports - replace with actual implementations
# from sentence_transformers import SentenceTransformer
# from pinecone import Pinecone
# from elasticsearch import Elasticsearch

@dataclass
class QueryResult:
    """Resultado de una consulta de retrieval"""
    query_id: str
    query_text: str
    retrieved_docs: List[Dict[str, Any]]
    response_time: float
    method: str
    
@dataclass
class BenchmarkResult:
    """Resultado de benchmark completo"""
    method: str
    precision_at_1: float
    precision_at_5: float
    recall_at_5: float
    ndcg_at_5: float
    avg_response_time: float
    total_queries: int

class RetrievalBenchmark:
    """Suite de benchmark para evaluar sistemas de retrieval"""
    
    def __init__(self, dataset_path: str, config: Dict[str, Any]):
        self.dataset_path = Path(dataset_path)
        self.config = config
        self.test_queries = self._load_test_dataset()
        self.results = []
        
    def _load_test_dataset(self) -> List[Dict[str, Any]]:
        """Carga dataset de prueba con queries y ground truth"""
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} test queries from {self.dataset_path}")
        return data
    
    def _calculate_precision_at_k(self, retrieved_ids: List[str], 
                                 relevant_ids: List[str], k: int) -> float:
        """Calcula Precision@K"""
        if not retrieved_ids or k == 0:
            return 0.0
            
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len([doc_id for doc_id in retrieved_at_k 
                                if doc_id in relevant_ids])
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    def _calculate_recall_at_k(self, retrieved_ids: List[str], 
                              relevant_ids: List[str], k: int) -> float:
        """Calcula Recall@K"""
        if not relevant_ids:
            return 0.0
            
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len([doc_id for doc_id in retrieved_at_k 
                                if doc_id in relevant_ids])
        return relevant_retrieved / len(relevant_ids)
    
    def _calculate_ndcg_at_k(self, retrieved_ids: List[str], 
                           relevant_ids: List[str], k: int) -> float:
        """Calcula NDCG@K - Normalized Discounted Cumulative Gain"""
        def dcg_at_k(relevances: List[float], k: int) -> float:
            relevances = relevances[:k]
            dcg = relevances[0] if relevances else 0.0
            for i in range(1, len(relevances)):
                dcg += relevances[i] / np.log2(i + 1)
            return dcg
        
        # Crear lista de relevancias (1 si es relevante, 0 si no)
        retrieved_at_k = retrieved_ids[:k]
        relevances = [1.0 if doc_id in relevant_ids else 0.0 
                     for doc_id in retrieved_at_k]
        
        # DCG actual
        dcg = dcg_at_k(relevances, k)
        
        # IDCG (mejor caso posible)
        ideal_relevances = [1.0] * min(len(relevant_ids), k) + [0.0] * max(0, k - len(relevant_ids))
        idcg = dcg_at_k(ideal_relevances, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def run_bm25_baseline(self) -> List[QueryResult]:
        """Ejecuta baseline BM25 (simulado)"""
        print("Running BM25 baseline...")
        results = []
        
        for query_data in self.test_queries:
            start_time = time.time()
            
            # Simulación de BM25 search
            # En implementación real: usar Elasticsearch o similar
            retrieved_docs = self._simulate_bm25_search(query_data["query"])
            
            response_time = time.time() - start_time
            
            result = QueryResult(
                query_id=query_data["id"],
                query_text=query_data["query"],
                retrieved_docs=retrieved_docs,
                response_time=response_time,
                method="bm25"
            )
            results.append(result)
            
        return results
    
    def run_semantic_search(self) -> List[QueryResult]:
        """Ejecuta búsqueda semántica con embeddings"""
        print("Running semantic search...")
        results = []
        
        # En implementación real: cargar modelo de embeddings
        # model = SentenceTransformer('all-MiniLM-L6-v2')
        
        for query_data in self.test_queries:
            start_time = time.time()
            
            # Simulación de semantic search
            retrieved_docs = self._simulate_semantic_search(query_data["query"])
            
            response_time = time.time() - start_time
            
            result = QueryResult(
                query_id=query_data["id"],
                query_text=query_data["query"],
                retrieved_docs=retrieved_docs,
                response_time=response_time,
                method="semantic"
            )
            results.append(result)
            
        return results
    
    def run_hybrid_search(self, bm25_weight: float = 0.3, 
                         semantic_weight: float = 0.7) -> List[QueryResult]:
        """Ejecuta búsqueda híbrida combinando BM25 y semántica"""
        print(f"Running hybrid search (BM25: {bm25_weight}, Semantic: {semantic_weight})...")
        results = []
        
        for query_data in self.test_queries:
            start_time = time.time()
            
            # Simulación de hybrid search
            retrieved_docs = self._simulate_hybrid_search(
                query_data["query"], bm25_weight, semantic_weight
            )
            
            response_time = time.time() - start_time
            
            result = QueryResult(
                query_id=query_data["id"],
                query_text=query_data["query"],
                retrieved_docs=retrieved_docs,
                response_time=response_time,
                method="hybrid"
            )
            results.append(result)
            
        return results
    
    def _simulate_bm25_search(self, query: str) -> List[Dict[str, Any]]:
        """Simula resultados de BM25 - reemplazar con implementación real"""
        # Simulación: BM25 es bueno para matches exactos pero limitado semánticamente
        import random
        random.seed(hash(query) % 1000)  # Deterministic based on query
        
        # Simular scores BM25 con bias hacia keywords
        docs = []
        for i in range(10):
            score = random.uniform(0.3, 0.9) if any(word in query.lower() 
                                                   for word in ["policy", "vacation", "remote"]) else random.uniform(0.1, 0.6)
            docs.append({
                "id": f"doc_{i}",
                "score": score,
                "title": f"Document {i}",
                "content": f"Sample content for document {i}"
            })
        
        return sorted(docs, key=lambda x: x["score"], reverse=True)
    
    def _simulate_semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Simula resultados de búsqueda semántica"""
        import random
        random.seed(hash(query) % 1000 + 1)  # Different seed than BM25
        
        # Simulación: Semántica es mejor para conceptos pero puede ser menos precisa
        docs = []
        for i in range(10):
            # Semantic search tiene distribución diferente
            score = random.uniform(0.4, 0.95) if len(query.split()) > 3 else random.uniform(0.2, 0.8)
            docs.append({
                "id": f"doc_{i}",
                "score": score,
                "title": f"Document {i}",
                "content": f"Sample content for document {i}"
            })
        
        return sorted(docs, key=lambda x: x["score"], reverse=True)
    
    def _simulate_hybrid_search(self, query: str, bm25_weight: float, 
                              semantic_weight: float) -> List[Dict[str, Any]]:
        """Simula resultados de búsqueda híbrida"""
        bm25_results = self._simulate_bm25_search(query)
        semantic_results = self._simulate_semantic_search(query)
        
        # Combinar scores
        combined_docs = {}
        
        for doc in bm25_results:
            doc_id = doc["id"]
            combined_docs[doc_id] = doc.copy()
            combined_docs[doc_id]["combined_score"] = doc["score"] * bm25_weight
            combined_docs[doc_id]["bm25_score"] = doc["score"]
            combined_docs[doc_id]["semantic_score"] = 0.0
        
        for doc in semantic_results:
            doc_id = doc["id"]
            if doc_id in combined_docs:
                combined_docs[doc_id]["combined_score"] += doc["score"] * semantic_weight
                combined_docs[doc_id]["semantic_score"] = doc["score"]
            else:
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["combined_score"] = doc["score"] * semantic_weight
                combined_docs[doc_id]["bm25_score"] = 0.0
                combined_docs[doc_id]["semantic_score"] = doc["score"]
        
        # Ordenar por score combinado
        result_docs = list(combined_docs.values())
        return sorted(result_docs, key=lambda x: x["combined_score"], reverse=True)
    
    def evaluate_results(self, query_results: List[QueryResult]) -> BenchmarkResult:
        """Evalúa resultados contra ground truth"""
        precisions_1 = []
        precisions_5 = []
        recalls_5 = []
        ndcgs_5 = []
        response_times = []
        
        for result in query_results:
            # Buscar ground truth para esta query
            ground_truth = None
            for query_data in self.test_queries:
                if query_data["id"] == result.query_id:
                    ground_truth = query_data
                    break
            
            if not ground_truth:
                continue
            
            # Extraer IDs de documentos recuperados
            retrieved_ids = [doc["id"] for doc in result.retrieved_docs]
            relevant_ids = ground_truth.get("relevant_docs", [])
            
            # Calcular métricas
            p1 = self._calculate_precision_at_k(retrieved_ids, relevant_ids, 1)
            p5 = self._calculate_precision_at_k(retrieved_ids, relevant_ids, 5)
            r5 = self._calculate_recall_at_k(retrieved_ids, relevant_ids, 5)
            ndcg5 = self._calculate_ndcg_at_k(retrieved_ids, relevant_ids, 5)
            
            precisions_1.append(p1)
            precisions_5.append(p5)
            recalls_5.append(r5)
            ndcgs_5.append(ndcg5)
            response_times.append(result.response_time)
        
        return BenchmarkResult(
            method=query_results[0].method if query_results else "unknown",
            precision_at_1=np.mean(precisions_1),
            precision_at_5=np.mean(precisions_5),
            recall_at_5=np.mean(recalls_5),
            ndcg_at_5=np.mean(ndcgs_5),
            avg_response_time=np.mean(response_times),
            total_queries=len(query_results)
        )
    
    def run_full_benchmark(self) -> Dict[str, BenchmarkResult]:
        """Ejecuta benchmark completo para todos los métodos"""
        print("Starting full retrieval benchmark...")
        
        # Ejecutar cada método
        bm25_results = self.run_bm25_baseline()
        semantic_results = self.run_semantic_search()
        hybrid_results = self.run_hybrid_search()
        
        # Evaluar resultados
        benchmark_results = {
            "bm25": self.evaluate_results(bm25_results),
            "semantic": self.evaluate_results(semantic_results),
            "hybrid": self.evaluate_results(hybrid_results)
        }
        
        return benchmark_results
    
    def generate_report(self, results: Dict[str, BenchmarkResult], 
                       output_path: Optional[str] = None) -> str:
        """Genera reporte de resultados"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if output_path is None:
            output_path = f"benchmark_report_{timestamp}.html"
        
        # Crear DataFrame para visualización
        data = []
        for method, result in results.items():
            data.append({
                "Method": method.upper(),
                "Precision@1": f"{result.precision_at_1:.3f}",
                "Precision@5": f"{result.precision_at_5:.3f}",
                "Recall@5": f"{result.recall_at_5:.3f}",
                "NDCG@5": f"{result.ndcg_at_5:.3f}",
                "Avg Response Time": f"{result.avg_response_time:.3f}s",
                "Total Queries": result.total_queries
            })
        
        df = pd.DataFrame(data)
        
        # Generar reporte HTML
        html_report = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Retrieval Benchmark Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                .winner {{ background-color: #d4edda; font-weight: bold; }}
                .metric {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Retrieval Benchmark Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Dataset:</strong> {self.dataset_path}</p>
            <p><strong>Total Queries:</strong> {len(self.test_queries)}</p>
            
            <h2>Results Summary</h2>
            {df.to_html(index=False, classes='benchmark-table')}
            
            <h2>Analysis</h2>
            <div class="metric">
                <h3>Best Performing Method</h3>
                <p>Based on NDCG@5 (overall relevance): <strong>{max(results.items(), key=lambda x: x[1].ndcg_at_5)[0].upper()}</strong></p>
                <p>Based on Response Time: <strong>{min(results.items(), key=lambda x: x[1].avg_response_time)[0].upper()}</strong></p>
            </div>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Use <strong>Hybrid</strong> approach for best relevance if computational resources allow</li>
                <li>Use <strong>BM25</strong> for low-latency requirements</li>
                <li>Use <strong>Semantic</strong> for complex conceptual queries</li>
            </ul>
        </body>
        </html>
        """
        
        # Escribir reporte
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        print(f"Benchmark report generated: {output_path}")
        return output_path
    
    def print_summary(self, results: Dict[str, BenchmarkResult]):
        """Imprime resumen de resultados en consola"""
        print("\n" + "="*60)
        print("RETRIEVAL BENCHMARK RESULTS")
        print("="*60)
        
        for method, result in results.items():
            print(f"\n{method.upper()} Results:")
            print(f"  Precision@1: {result.precision_at_1:.3f}")
            print(f"  Precision@5: {result.precision_at_5:.3f}")
            print(f"  Recall@5: {result.recall_at_5:.3f}")
            print(f"  NDCG@5: {result.ndcg_at_5:.3f}")
            print(f"  Avg Response Time: {result.avg_response_time:.3f}s")
        
        # Determinar ganador
        best_method = max(results.items(), key=lambda x: x[1].ndcg_at_5)
        print(f"\n🏆 WINNER: {best_method[0].upper()} with NDCG@5 of {best_method[1].ndcg_at_5:.3f}")

def create_sample_dataset():
    """Crea dataset de ejemplo para testing"""
    sample_data = [
        {
            "id": "q001",
            "query": "How many vacation days do I get after 2 years?",
            "relevant_docs": ["doc_1", "doc_3", "doc_7"]
        },
        {
            "id": "q002", 
            "query": "Remote work policy requirements",
            "relevant_docs": ["doc_2", "doc_5", "doc_8"]
        },
        {
            "id": "q003",
            "query": "Security compliance for data handling",
            "relevant_docs": ["doc_4", "doc_6", "doc_9"]
        }
    ] * 10  # Repeat for larger dataset
    
    # Update IDs to be unique
    for i, item in enumerate(sample_data):
        item["id"] = f"q{i+1:03d}"
    
    with open("sample_test_dataset.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("Sample dataset created: sample_test_dataset.json")

def main():
    """Función principal para ejecutar benchmark"""
    # Crear dataset de ejemplo si no existe
    if not Path("sample_test_dataset.json").exists():
        create_sample_dataset()
    
    # Configuración
    config = {
        "models": {
            "embedding_model": "all-MiniLM-L6-v2",
            "rerank_model": "cross-encoder/ms-marco-MiniLM-L-2-v2"
        },
        "search_params": {
            "top_k": 10,
            "similarity_threshold": 0.7
        }
    }
    
    # Ejecutar benchmark
    benchmark = RetrievalBenchmark("sample_test_dataset.json", config)
    results = benchmark.run_full_benchmark()
    
    # Mostrar resultados
    benchmark.print_summary(results)
    
    # Generar reporte
    report_path = benchmark.generate_report(results)
    print(f"\nDetailed report available at: {report_path}")

if __name__ == "__main__":
    main()
