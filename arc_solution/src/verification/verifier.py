"""
ARC Prize 2025 - Hypothesis Verifier

This module implements verification and ranking of hypotheses based on
multiple criteria including accuracy, consistency, and complexity.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ..core.types import Grid, Task, Hypothesis
from ..execution.executor import ExecutionEngine, ExecutionResult


@dataclass
class VerificationResult:
    """Result of hypothesis verification"""
    hypothesis: Hypothesis
    score: float
    accuracy: float
    consistency: float
    complexity_penalty: float
    execution_success_rate: float
    metadata: Dict[str, Any]


class HypothesisVerifier:
    """
    Verifies and ranks hypotheses based on multiple criteria:
    
    1. Accuracy: How well it matches training examples
    2. Consistency: How reliable the execution is
    3. Complexity: Preference for simpler solutions
    4. Generalization: How well it might generalize
    """
    
    def __init__(self, executor: ExecutionEngine):
        self.executor = executor
        
        # Scoring weights
        self.accuracy_weight = 0.6
        self.consistency_weight = 0.2
        self.complexity_weight = 0.1
        self.generalization_weight = 0.1
        
        # Thresholds
        self.min_accuracy_threshold = 0.8
        self.max_complexity_penalty = 0.5
    
    def verify_hypotheses(self, hypotheses: List[Hypothesis], task: Task) -> List[VerificationResult]:
        """
        Verify and rank a list of hypotheses
        
        Args:
            hypotheses: List of hypotheses to verify
            task: The ARC task for verification
            
        Returns:
            List of verification results sorted by score (descending)
        """
        results = []
        
        for hypothesis in hypotheses:
            verification_result = self.verify_single_hypothesis(hypothesis, task)
            results.append(verification_result)
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results
    
    def verify_single_hypothesis(self, hypothesis: Hypothesis, task: Task) -> VerificationResult:
        """Verify a single hypothesis"""
        try:
            # Compute accuracy score
            accuracy_result = self._compute_accuracy(hypothesis, task)
            accuracy = accuracy_result['accuracy']
            execution_success_rate = accuracy_result['execution_success_rate']
            
            # Compute consistency score
            consistency = self._compute_consistency(hypothesis, task)
            
            # Compute complexity penalty
            complexity_penalty = self._compute_complexity_penalty(hypothesis)
            
            # Compute generalization score
            generalization = self._compute_generalization_score(hypothesis, task)
            
            # Compute overall score
            score = (
                self.accuracy_weight * accuracy +
                self.consistency_weight * consistency +
                self.complexity_weight * (1.0 - complexity_penalty) +
                self.generalization_weight * generalization
            )
            
            # Apply minimum accuracy threshold
            if accuracy < self.min_accuracy_threshold:
                score *= 0.1  # Severely penalize low accuracy
            
            return VerificationResult(
                hypothesis=hypothesis,
                score=score,
                accuracy=accuracy,
                consistency=consistency,
                complexity_penalty=complexity_penalty,
                execution_success_rate=execution_success_rate,
                metadata={
                    'generalization_score': generalization,
                    'detailed_accuracy': accuracy_result
                }
            )
        
        except Exception as e:
            # Return low-score result for failed verification
            return VerificationResult(
                hypothesis=hypothesis,
                score=0.0,
                accuracy=0.0,
                consistency=0.0,
                complexity_penalty=1.0,
                execution_success_rate=0.0,
                metadata={'verification_error': str(e)}
            )
    
    def _compute_accuracy(self, hypothesis: Hypothesis, task: Task) -> Dict[str, Any]:
        """Compute accuracy of hypothesis on training examples"""
        if not hypothesis.program or not task.train_pairs:
            return {
                'accuracy': 0.0,
                'execution_success_rate': 0.0,
                'matches': 0,
                'total': len(task.train_pairs),
                'execution_failures': 0
            }
        
        matches = 0
        execution_failures = 0
        total = len(task.train_pairs)
        
        for pair in task.train_pairs:
            input_grid = pair['input']
            expected_output = pair['output']
            
            # Execute hypothesis
            execution_result = self.executor.execute_program(hypothesis.program, input_grid)
            
            if execution_result.success and execution_result.output:
                # Check if output matches expected
                if self._grids_equal(execution_result.output, expected_output):
                    matches += 1
            else:
                execution_failures += 1
        
        accuracy = matches / total if total > 0 else 0.0
        execution_success_rate = (total - execution_failures) / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'execution_success_rate': execution_success_rate,
            'matches': matches,
            'total': total,
            'execution_failures': execution_failures
        }
    
    def _compute_consistency(self, hypothesis: Hypothesis, task: Task) -> float:
        """Compute consistency of hypothesis execution"""
        if not hypothesis.program or not task.train_pairs:
            return 0.0
        
        execution_times = []
        execution_successes = []
        
        # Run multiple times to check consistency
        for pair in task.train_pairs:
            input_grid = pair['input']
            
            # Execute multiple times
            for _ in range(3):  # 3 executions per example
                execution_result = self.executor.execute_program(hypothesis.program, input_grid)
                execution_times.append(execution_result.execution_time)
                execution_successes.append(execution_result.success)
        
        if not execution_times:
            return 0.0
        
        # Consistency based on execution time variance and success rate
        time_variance = np.var(execution_times) if len(execution_times) > 1 else 0.0
        success_rate = sum(execution_successes) / len(execution_successes)
        
        # Normalize time variance (lower is better)
        max_time = max(execution_times) if execution_times else 1.0
        normalized_time_variance = min(time_variance / max_time, 1.0) if max_time > 0 else 0.0
        
        # Consistency is high when success rate is high and time variance is low
        consistency = success_rate * (1.0 - normalized_time_variance)
        
        return consistency
    
    def _compute_complexity_penalty(self, hypothesis: Hypothesis) -> float:
        """Compute complexity penalty (0.0 = simple, 1.0 = very complex)"""
        if not hypothesis.program:
            return 0.0
        
        operations_count = len(hypothesis.program.operations)
        max_operations = 10  # Assume 10 operations is very complex
        
        # Basic complexity from operation count
        operation_complexity = min(operations_count / max_operations, 1.0)
        
        # Parameter complexity (operations with many parameters are more complex)
        parameter_complexity = 0.0
        for operation in hypothesis.program.operations:
            param_count = len([k for k in operation.keys() if k != 'operation'])
            parameter_complexity += param_count / 5.0  # Assume 5 params is complex
        
        parameter_complexity = min(parameter_complexity / operations_count, 1.0) if operations_count > 0 else 0.0
        
        # Combined complexity
        total_complexity = 0.7 * operation_complexity + 0.3 * parameter_complexity
        
        return min(total_complexity, 1.0)
    
    def _compute_generalization_score(self, hypothesis: Hypothesis, task: Task) -> float:
        """Compute how well hypothesis might generalize"""
        if not hypothesis.program or not task.train_pairs:
            return 0.0
        
        # Factors that suggest good generalization:
        # 1. Works on multiple training examples consistently
        # 2. Simple transformations (geometric, color mapping)
        # 3. Doesn't rely on specific grid sizes or positions
        
        # Check if it works on all training examples
        accuracy_result = self._compute_accuracy(hypothesis, task)
        base_generalization = accuracy_result['accuracy']
        
        # Bonus for simple, interpretable operations
        operation_bonus = 0.0
        simple_operations = {'rotate', 'reflect', 'translate', 'recolor', 'scale'}
        
        for operation in hypothesis.program.operations:
            op_name = operation.get('operation', '')
            if op_name in simple_operations:
                operation_bonus += 0.1
        
        operation_bonus = min(operation_bonus, 0.3)  # Cap at 0.3
        
        # Penalty for operations that might be too specific
        specificity_penalty = 0.0
        specific_operations = {'paint_pattern', 'crop', 'pad'}
        
        for operation in hypothesis.program.operations:
            op_name = operation.get('operation', '')
            if op_name in specific_operations:
                specificity_penalty += 0.1
        
        specificity_penalty = min(specificity_penalty, 0.2)  # Cap at 0.2
        
        generalization = base_generalization + operation_bonus - specificity_penalty
        
        return max(0.0, min(1.0, generalization))
    
    def _grids_equal(self, grid1: Grid, grid2: Grid) -> bool:
        """Check if two grids are exactly equal"""
        try:
            if len(grid1) != len(grid2):
                return False
            
            for row1, row2 in zip(grid1, grid2):
                if len(row1) != len(row2):
                    return False
                
                for cell1, cell2 in zip(row1, row2):
                    if cell1 != cell2:
                        return False
            
            return True
        
        except:
            return False
    
    def select_best_hypotheses(self, verification_results: List[VerificationResult], 
                              max_count: int = 3) -> List[Hypothesis]:
        """
        Select the best hypotheses from verification results
        
        Args:
            verification_results: List of verification results (should be sorted)
            max_count: Maximum number of hypotheses to return
            
        Returns:
            List of best hypotheses
        """
        # Filter out hypotheses with very low scores
        min_score_threshold = 0.1
        valid_results = [r for r in verification_results if r.score >= min_score_threshold]
        
        # Select top hypotheses
        selected_results = valid_results[:max_count]
        
        return [result.hypothesis for result in selected_results]
    
    def get_verification_summary(self, verification_results: List[VerificationResult]) -> Dict[str, Any]:
        """Get summary statistics of verification results"""
        if not verification_results:
            return {}
        
        scores = [r.score for r in verification_results]
        accuracies = [r.accuracy for r in verification_results]
        consistencies = [r.consistency for r in verification_results]
        
        return {
            'total_hypotheses': len(verification_results),
            'best_score': max(scores),
            'worst_score': min(scores),
            'avg_score': sum(scores) / len(scores),
            'best_accuracy': max(accuracies),
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'best_consistency': max(consistencies),
            'avg_consistency': sum(consistencies) / len(consistencies),
            'high_quality_count': len([r for r in verification_results if r.score >= 0.8]),
            'medium_quality_count': len([r for r in verification_results if 0.5 <= r.score < 0.8]),
            'low_quality_count': len([r for r in verification_results if r.score < 0.5])
        }
