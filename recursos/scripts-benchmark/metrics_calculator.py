#!/usr/bin/env python3
"""
Metrics Calculator para Portal 4
Calcula métricas básicas de evaluación sin dependencias externas
"""

import json
import math
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

@dataclass
class MetricResult:
    """Resultado de una métrica calculada"""
    name: str
    value: float
    description: str
    higher_is_better: bool = True

class MetricsCalculator:
    """Calculadora de métricas para evaluación de modelos y sistemas"""
    
    @staticmethod
    def precision_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """Calcula Precision@K"""
        if k <= 0 or not retrieved:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_at_k if item in relevant)
        
        return relevant_retrieved / min(k, len(retrieved_at_k))
    
    @staticmethod
    def recall_at_k(retrieved: List[Any], relevant: List[Any], k: int) -> float:
        """Calcula Recall@K"""
        if not relevant or k <= 0:
            return 0.0
        
        retrieved_at_k = retrieved[:k]
        relevant_retrieved = sum(1 for item in retrieved_at_k if item in relevant)
        
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def f1_score(precision: float, recall: float) -> float:
        """Calcula F1 Score"""
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def dcg_at_k(relevances: List[float], k: int) -> float:
        """Calcula Discounted Cumulative Gain at K"""
        if k <= 0 or not relevances:
            return 0.0
        
        relevances = relevances[:k]
        dcg = relevances[0] if relevances else 0.0
        
        for i in range(1, len(relevances)):
            dcg += relevances[i] / math.log2(i + 1)
        
        return dcg
    
    @staticmethod
    def ndcg_at_k(retrieved_relevances: List[float], ideal_relevances: List[float], k: int) -> float:
        """Calcula Normalized Discounted Cumulative Gain at K"""
        dcg = MetricsCalculator.dcg_at_k(retrieved_relevances, k)
        idcg = MetricsCalculator.dcg_at_k(sorted(ideal_relevances, reverse=True), k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def mean_reciprocal_rank(rankings: List[List[Any]], relevant_items: List[List[Any]]) -> float:
        """Calcula Mean Reciprocal Rank"""
        if not rankings or not relevant_items:
            return 0.0
        
        reciprocal_ranks = []
        
        for ranking, relevant in zip(rankings, relevant_items):
            rr = 0.0
            for i, item in enumerate(ranking):
                if item in relevant:
                    rr = 1.0 / (i + 1)
                    break
            reciprocal_ranks.append(rr)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    @staticmethod
    def jaccard_similarity(set1: set, set2: set) -> float:
        """Calcula similitud de Jaccard"""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calcula similitud coseno entre dos vectores"""
        if len(vec1) != len(vec2):
            raise ValueError("Vectors must have the same length")
        
        if not vec1 or not vec2:
            return 0.0
        
        # Producto punto
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    @staticmethod
    def bleu_score_simple(candidate: str, reference: str, n: int = 4) -> float:
        """Calcula BLEU score simplificado (solo n-gramas, sin brevity penalty)"""
        def get_ngrams(text: str, n: int) -> List[str]:
            words = text.lower().split()
            return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not candidate.strip() or not reference.strip():
            return 0.0
        
        scores = []
        
        for i in range(1, n + 1):
            candidate_ngrams = get_ngrams(candidate, i)
            reference_ngrams = get_ngrams(reference, i)
            
            if not candidate_ngrams:
                scores.append(0.0)
                continue
            
            # Contar matches
            matches = 0
            ref_counts = {}
            for ngram in reference_ngrams:
                ref_counts[ngram] = ref_counts.get(ngram, 0) + 1
            
            for ngram in candidate_ngrams:
                if ngram in ref_counts and ref_counts[ngram] > 0:
                    matches += 1
                    ref_counts[ngram] -= 1
            
            precision = matches / len(candidate_ngrams)
            scores.append(precision)
        
        # Promedio geométrico
        if any(score == 0 for score in scores):
            return 0.0
        
        geometric_mean = 1.0
        for score in scores:
            geometric_mean *= score
        
        return geometric_mean ** (1.0 / len(scores))
    
    @staticmethod
    def rouge_l_score(candidate: str, reference: str) -> Dict[str, float]:
        """Calcula ROUGE-L score (Longest Common Subsequence)"""
        def lcs_length(x: List[str], y: List[str]) -> int:
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        candidate_words = candidate.lower().split()
        reference_words = reference.lower().split()
        
        if not candidate_words or not reference_words:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        
        lcs_len = lcs_length(candidate_words, reference_words)
        
        precision = lcs_len / len(candidate_words)
        recall = lcs_len / len(reference_words)
        f1 = MetricsCalculator.f1_score(precision, recall)
        
        return {"precision": precision, "recall": recall, "f1": f1}
    
    @staticmethod
    def accuracy(predictions: List[Any], ground_truth: List[Any]) -> float:
        """Calcula accuracy simple"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        if not predictions:
            return 0.0
        
        correct = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
        return correct / len(predictions)
    
    @staticmethod
    def mean_absolute_error(predictions: List[float], ground_truth: List[float]) -> float:
        """Calcula Mean Absolute Error"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        if not predictions:
            return 0.0
        
        errors = [abs(pred - truth) for pred, truth in zip(predictions, ground_truth)]
        return sum(errors) / len(errors)
    
    @staticmethod
    def mean_squared_error(predictions: List[float], ground_truth: List[float]) -> float:
        """Calcula Mean Squared Error"""
        if len(predictions) != len(ground_truth):
            raise ValueError("Predictions and ground truth must have the same length")
        
        if not predictions:
            return 0.0
        
        errors = [(pred - truth) ** 2 for pred, truth in zip(predictions, ground_truth)]
        return sum(errors) / len(errors)
    
    @staticmethod
    def root_mean_squared_error(predictions: List[float], ground_truth: List[float]) -> float:
        """Calcula Root Mean Squared Error"""
        mse = MetricsCalculator.mean_squared_error(predictions, ground_truth)
        return math.sqrt(mse)

class BenchmarkEvaluator:
    """Evaluador que combina múltiples métricas"""
    
    def __init__(self):
        self.metrics = MetricsCalculator()
    
    def evaluate_retrieval_system(self, results: List[Dict[str, Any]]) -> Dict[str, MetricResult]:
        """Evalúa un sistema de retrieval"""
        all_metrics = {}
        
        # Calcular métricas agregadas
        precision_1_scores = []
        precision_5_scores = []
        recall_5_scores = []
        ndcg_5_scores = []
        
        for result in results:
            retrieved = result.get("retrieved", [])
            relevant = result.get("relevant", [])
            
            # Precision@K
            p1 = self.metrics.precision_at_k(retrieved, relevant, 1)
            p5 = self.metrics.precision_at_k(retrieved, relevant, 5)
            r5 = self.metrics.recall_at_k(retrieved, relevant, 5)
            
            precision_1_scores.append(p1)
            precision_5_scores.append(p5)
            recall_5_scores.append(r5)
            
            # NDCG (simulando relevances)
            retrieved_relevances = [1.0 if item in relevant else 0.0 for item in retrieved[:5]]
            ideal_relevances = [1.0] * min(len(relevant), 5)
            ndcg5 = self.metrics.ndcg_at_k(retrieved_relevances, ideal_relevances, 5)
            ndcg_5_scores.append(ndcg5)
        
        # Promediar métricas
        all_metrics["precision_at_1"] = MetricResult(
            "Precision@1", 
            sum(precision_1_scores) / len(precision_1_scores) if precision_1_scores else 0.0,
            "Precision at rank 1"
        )
        
        all_metrics["precision_at_5"] = MetricResult(
            "Precision@5",
            sum(precision_5_scores) / len(precision_5_scores) if precision_5_scores else 0.0,
            "Precision at rank 5"
        )
        
        all_metrics["recall_at_5"] = MetricResult(
            "Recall@5",
            sum(recall_5_scores) / len(recall_5_scores) if recall_5_scores else 0.0,
            "Recall at rank 5"
        )
        
        all_metrics["ndcg_at_5"] = MetricResult(
            "NDCG@5",
            sum(ndcg_5_scores) / len(ndcg_5_scores) if ndcg_5_scores else 0.0,
            "Normalized Discounted Cumulative Gain at rank 5"
        )
        
        return all_metrics
    
    def evaluate_generation_system(self, results: List[Dict[str, Any]]) -> Dict[str, MetricResult]:
        """Evalúa un sistema de generación de texto"""
        all_metrics = {}
        
        bleu_scores = []
        rouge_f1_scores = []
        
        for result in results:
            generated = result.get("generated", "")
            reference = result.get("reference", "")
            
            if generated and reference:
                # BLEU
                bleu = self.metrics.bleu_score_simple(generated, reference)
                bleu_scores.append(bleu)
                
                # ROUGE-L
                rouge = self.metrics.rouge_l_score(generated, reference)
                rouge_f1_scores.append(rouge["f1"])
        
        all_metrics["bleu"] = MetricResult(
            "BLEU",
            sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0,
            "Bilingual Evaluation Understudy score"
        )
        
        all_metrics["rouge_l"] = MetricResult(
            "ROUGE-L F1",
            sum(rouge_f1_scores) / len(rouge_f1_scores) if rouge_f1_scores else 0.0,
            "ROUGE-L F1 score"
        )
        
        return all_metrics
    
    def evaluate_classification_system(self, predictions: List[Any], 
                                     ground_truth: List[Any]) -> Dict[str, MetricResult]:
        """Evalúa un sistema de clasificación"""
        all_metrics = {}
        
        # Accuracy
        acc = self.metrics.accuracy(predictions, ground_truth)
        all_metrics["accuracy"] = MetricResult(
            "Accuracy",
            acc,
            "Classification accuracy"
        )
        
        return all_metrics
    
    def print_metrics_report(self, metrics: Dict[str, MetricResult], title: str = "Metrics Report"):
        """Imprime reporte de métricas"""
        print(f"\n{'='*60}")
        print(f"{title.upper()}")
        print(f"{'='*60}")
        
        for metric_name, metric in metrics.items():
            direction = "↑" if metric.higher_is_better else "↓"
            print(f"{metric.name}: {metric.value:.4f} {direction}")
            print(f"  {metric.description}")
            print()

def demo_metrics():
    """Demonstración de uso de métricas"""
    print("🧮 Portal 4 - Metrics Calculator Demo")
    
    calc = MetricsCalculator()
    evaluator = BenchmarkEvaluator()
    
    # Demo 1: Retrieval metrics
    print("\n📊 Retrieval Metrics Demo")
    retrieved_docs = ["doc1", "doc3", "doc5", "doc2", "doc7"]
    relevant_docs = ["doc1", "doc2", "doc4"]
    
    p1 = calc.precision_at_k(retrieved_docs, relevant_docs, 1)
    p5 = calc.precision_at_k(retrieved_docs, relevant_docs, 5)
    r5 = calc.recall_at_k(retrieved_docs, relevant_docs, 5)
    
    print(f"Precision@1: {p1:.3f}")
    print(f"Precision@5: {p5:.3f}")
    print(f"Recall@5: {r5:.3f}")
    print(f"F1@5: {calc.f1_score(p5, r5):.3f}")
    
    # Demo 2: Text similarity
    print("\n📝 Text Similarity Demo")
    text1 = "The quick brown fox jumps over the lazy dog"
    text2 = "A quick brown fox leaps over a lazy dog"
    
    rouge = calc.rouge_l_score(text1, text2)
    bleu = calc.bleu_score_simple(text1, text2)
    
    print(f"ROUGE-L F1: {rouge['f1']:.3f}")
    print(f"BLEU: {bleu:.3f}")
    
    # Demo 3: Classification metrics
    print("\n🎯 Classification Demo")
    predictions = ["A", "B", "A", "C", "B", "A"]
    ground_truth = ["A", "B", "C", "C", "B", "A"]
    
    accuracy = calc.accuracy(predictions, ground_truth)
    print(f"Accuracy: {accuracy:.3f}")
    
    # Demo 4: Numerical predictions
    print("\n📈 Regression Metrics Demo")
    pred_values = [2.5, 3.1, 4.2, 1.8, 2.9]
    true_values = [2.3, 3.0, 4.5, 2.0, 2.8]
    
    mae = calc.mean_absolute_error(pred_values, true_values)
    mse = calc.mean_squared_error(pred_values, true_values)
    rmse = calc.root_mean_squared_error(pred_values, true_values)
    
    print(f"MAE: {mae:.3f}")
    print(f"MSE: {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")

def create_metrics_config():
    """Crea archivo de configuración para métricas"""
    config = {
        "retrieval_metrics": {
            "precision_at_k": [1, 3, 5, 10],
            "recall_at_k": [5, 10],
            "ndcg_at_k": [5, 10],
            "mrr": True
        },
        "generation_metrics": {
            "bleu": {"n_grams": [1, 2, 3, 4]},
            "rouge": ["rouge-1", "rouge-2", "rouge-l"],
            "semantic_similarity": True
        },
        "classification_metrics": {
            "accuracy": True,
            "precision": True,
            "recall": True,
            "f1": True
        },
        "regression_metrics": {
            "mae": True,
            "mse": True,
            "rmse": True,
            "r2": True
        }
    }
    
    with open("metrics_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("Metrics configuration created: metrics_config.json")

def main():
    """Función principal"""
    print("🧮 Portal 4 - Metrics Calculator")
    print("="*50)
    
    # Crear configuración si no existe
    create_metrics_config()
    
    # Ejecutar demo
    demo_metrics()
    
    print("\n✅ Metrics calculator ready for use!")
    print("Usage example:")
    print("  from metrics_calculator import MetricsCalculator")
    print("  calc = MetricsCalculator()")
    print("  precision = calc.precision_at_k(retrieved, relevant, 5)")

if __name__ == "__main__":
    main()
