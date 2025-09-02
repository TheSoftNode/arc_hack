"""
ARC Prize 2025 - Main Pipeline

This is the core pipeline that orchestrates all components of our
multi-agent neuro-symbolic reasoning system.
"""

import time
import logging
from typing import List, Dict, Any
from pathlib import Path

from .types import Task, TaskSolution, Hypothesis, Grid
from ..hypothesis_generators.symbolic_solver import SymbolicSolver
from ..hypothesis_generators.llm_reasoner import LLMReasoner
from ..hypothesis_generators.vision_solver import VisionSolver
from ..execution_engine.executor import HypothesisExecutor
from ..execution_engine.verifier import SolutionVerifier
from .preprocessor import MultiModalPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARCReasoningPipeline:
    """
    Main pipeline for ARC task solving.
    
    Architecture:
    Input Task → [Preprocessor] → [Multi-Hypothesis Generator] → [Execution & Verification] → Output
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the reasoning pipeline with configuration"""
        self.config = config
        
        # Initialize components
        self.preprocessor = MultiModalPreprocessor(config.get('preprocessor', {}))
        self.symbolic_solver = SymbolicSolver(config.get('symbolic', {}))
        self.llm_reasoner = LLMReasoner(config.get('llm', {}))
        self.vision_solver = VisionSolver(config.get('vision', {}))
        self.executor = HypothesisExecutor(config.get('executor', {}))
        self.verifier = SolutionVerifier(config.get('verifier', {}))
        
        # Performance tracking
        self.stats = {
            'tasks_solved': 0,
            'total_tasks': 0,
            'avg_execution_time': 0.0,
            'hypothesis_success_rate': {}
        }
        
    def solve_task(self, task: Task) -> TaskSolution:
        """
        Solve a single ARC task using multi-agent reasoning.
        
        Args:
            task: The ARC task to solve
            
        Returns:
            TaskSolution containing predictions and metadata
        """
        start_time = time.time()
        logger.info(f"Starting solution for task {task.task_id}")
        
        try:
            # Phase 1: Preprocessing & Multi-Modal Representation
            scene_representations = self._preprocess_task(task)
            
            # Phase 2: Multi-Hypothesis Generation
            hypotheses = self._generate_hypotheses(task, scene_representations)
            
            # Phase 3: Hypothesis Execution & Verification
            validated_hypotheses = self._validate_hypotheses(hypotheses, task)
            
            # Phase 4: Prediction Generation
            predictions = self._generate_predictions(validated_hypotheses, task)
            
            # Phase 5: Solution Assembly
            solution = self._assemble_solution(
                task, hypotheses, predictions, time.time() - start_time
            )
            
            self._update_stats(solution)
            logger.info(f"Completed task {task.task_id} in {solution.execution_time:.2f}s")
            
            return solution
            
        except Exception as e:
            logger.error(f"Error solving task {task.task_id}: {str(e)}")
            # Return fallback solution
            return self._create_fallback_solution(task, time.time() - start_time)
    
    def solve_batch(self, tasks: List[Task]) -> List[TaskSolution]:
        """Solve multiple tasks in batch"""
        solutions = []
        
        for task in tasks:
            solution = self.solve_task(task)
            solutions.append(solution)
            
        return solutions
    
    def _preprocess_task(self, task: Task) -> List[Any]:
        """Preprocess task into multi-modal representations"""
        logger.debug(f"Preprocessing task {task.task_id}")
        
        scene_representations = []
        
        # Process training pairs
        for i, pair in enumerate(task.train_pairs):
            input_scene = self.preprocessor.process_grid(pair['input'])
            output_scene = self.preprocessor.process_grid(pair['output'])
            scene_representations.append({
                'input': input_scene,
                'output': output_scene,
                'pair_id': i
            })
            
        # Process test inputs
        for i, test_input in enumerate(task.test_inputs):
            test_scene = self.preprocessor.process_grid(test_input)
            scene_representations.append({
                'test_input': test_scene,
                'test_id': i
            })
            
        return scene_representations
    
    def _generate_hypotheses(self, task: Task, scenes: List[Any]) -> List[Hypothesis]:
        """Generate hypotheses using multiple reasoning engines"""
        logger.debug(f"Generating hypotheses for task {task.task_id}")
        
        all_hypotheses = []
        
        # Symbolic reasoning hypotheses
        try:
            symbolic_hypotheses = self.symbolic_solver.generate_hypotheses(task, scenes)
            all_hypotheses.extend(symbolic_hypotheses)
            logger.debug(f"Generated {len(symbolic_hypotheses)} symbolic hypotheses")
        except Exception as e:
            logger.warning(f"Symbolic solver failed: {str(e)}")
        
        # LLM reasoning hypotheses
        try:
            llm_hypotheses = self.llm_reasoner.generate_hypotheses(task, scenes)
            all_hypotheses.extend(llm_hypotheses)
            logger.debug(f"Generated {len(llm_hypotheses)} LLM hypotheses")
        except Exception as e:
            logger.warning(f"LLM reasoner failed: {str(e)}")
        
        # Vision-based hypotheses
        try:
            vision_hypotheses = self.vision_solver.generate_hypotheses(task, scenes)
            all_hypotheses.extend(vision_hypotheses)
            logger.debug(f"Generated {len(vision_hypotheses)} vision hypotheses")
        except Exception as e:
            logger.warning(f"Vision solver failed: {str(e)}")
        
        # Sort by confidence
        all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
        
        logger.info(f"Generated {len(all_hypotheses)} total hypotheses")
        return all_hypotheses
    
    def _validate_hypotheses(self, hypotheses: List[Hypothesis], task: Task) -> List[Hypothesis]:
        """Validate hypotheses against training examples"""
        logger.debug(f"Validating {len(hypotheses)} hypotheses")
        
        validated = []
        
        for hypothesis in hypotheses:
            try:
                # Test hypothesis on training pairs
                is_valid = self.verifier.validate_hypothesis(hypothesis, task)
                
                if is_valid:
                    validated.append(hypothesis)
                    logger.debug(f"Hypothesis '{hypothesis.description}' validated successfully")
                else:
                    logger.debug(f"Hypothesis '{hypothesis.description}' failed validation")
                    
            except Exception as e:
                logger.warning(f"Error validating hypothesis: {str(e)}")
                continue
        
        logger.info(f"Validated {len(validated)} hypotheses")
        return validated
    
    def _generate_predictions(self, hypotheses: List[Hypothesis], task: Task) -> List[Dict[str, Grid]]:
        """Generate predictions for test inputs"""
        logger.debug(f"Generating predictions from {len(hypotheses)} hypotheses")
        
        predictions = []
        
        for test_input in task.test_inputs:
            prediction_attempts = {'attempt_1': None, 'attempt_2': None}
            
            # Use top hypotheses for attempts
            for i, hypothesis in enumerate(hypotheses[:2]):
                try:
                    prediction = self.executor.execute_hypothesis(hypothesis, test_input)
                    prediction_attempts[f'attempt_{i+1}'] = prediction
                except Exception as e:
                    logger.warning(f"Error executing hypothesis {i+1}: {str(e)}")
                    # Use fallback prediction
                    prediction_attempts[f'attempt_{i+1}'] = self._fallback_prediction(test_input)
            
            # Ensure we have two attempts
            if prediction_attempts['attempt_1'] is None:
                prediction_attempts['attempt_1'] = self._fallback_prediction(test_input)
            if prediction_attempts['attempt_2'] is None:
                prediction_attempts['attempt_2'] = self._fallback_prediction(test_input)
                
            predictions.append(prediction_attempts)
        
        return predictions
    
    def _assemble_solution(self, task: Task, hypotheses: List[Hypothesis], 
                          predictions: List[Dict[str, Grid]], execution_time: float) -> TaskSolution:
        """Assemble final solution"""
        
        # Calculate confidence scores
        confidence_scores = []
        for i, hypothesis in enumerate(hypotheses[:len(predictions)]):
            confidence_scores.append(hypothesis.confidence)
        
        # Pad with lower confidence if needed
        while len(confidence_scores) < len(predictions):
            confidence_scores.append(0.1)
            
        metadata = {
            'num_hypotheses_generated': len(hypotheses),
            'preprocessing_time': 0.0,  # TODO: Track this separately
            'reasoning_time': execution_time,
            'solution_method': 'multi_agent_neuro_symbolic'
        }
        
        return TaskSolution(
            task_id=task.task_id,
            hypotheses=hypotheses,
            predictions=predictions,
            confidence_scores=confidence_scores,
            execution_time=execution_time,
            metadata=metadata
        )
    
    def _create_fallback_solution(self, task: Task, execution_time: float) -> TaskSolution:
        """Create a fallback solution when main pipeline fails"""
        logger.warning(f"Creating fallback solution for task {task.task_id}")
        
        predictions = []
        for test_input in task.test_inputs:
            fallback_pred = self._fallback_prediction(test_input)
            predictions.append({
                'attempt_1': fallback_pred,
                'attempt_2': fallback_pred
            })
        
        return TaskSolution(
            task_id=task.task_id,
            hypotheses=[],
            predictions=predictions,
            confidence_scores=[0.01] * len(predictions),
            execution_time=execution_time,
            metadata={'solution_method': 'fallback'}
        )
    
    def _fallback_prediction(self, input_grid: Grid) -> Grid:
        """Generate a fallback prediction (common patterns)"""
        # Strategy 1: Return input unchanged
        if len(input_grid) <= 10:  # Small grids
            return input_grid
            
        # Strategy 2: Return grid of zeros
        return [[0 for _ in range(len(input_grid[0]))] for _ in range(len(input_grid))]
    
    def _update_stats(self, solution: TaskSolution):
        """Update pipeline statistics"""
        self.stats['total_tasks'] += 1
        if solution.confidence_scores and max(solution.confidence_scores) > 0.5:
            self.stats['tasks_solved'] += 1
            
        # Update average execution time
        old_avg = self.stats['avg_execution_time']
        old_count = self.stats['total_tasks'] - 1
        self.stats['avg_execution_time'] = (old_avg * old_count + solution.execution_time) / self.stats['total_tasks']
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        return {
            'success_rate': self.stats['tasks_solved'] / max(1, self.stats['total_tasks']),
            'total_tasks_processed': self.stats['total_tasks'],
            'average_execution_time': self.stats['avg_execution_time'],
            'component_stats': {
                'symbolic_solver': getattr(self.symbolic_solver, 'stats', {}),
                'llm_reasoner': getattr(self.llm_reasoner, 'stats', {}),
                'vision_solver': getattr(self.vision_solver, 'stats', {})
            }
        }


def create_default_config() -> Dict[str, Any]:
    """Create default configuration for the pipeline"""
    return {
        'preprocessor': {
            'enable_object_detection': True,
            'enable_graph_representation': True,
            'enable_feature_extraction': True
        },
        'symbolic': {
            'max_search_depth': 5,
            'timeout_seconds': 10,
            'use_z3_solver': True
        },
        'llm': {
            'model_name': 'deepseek-v3',
            'max_tokens': 2048,
            'temperature': 0.1,
            'use_caching': True
        },
        'vision': {
            'enable_morphological_ops': True,
            'enable_contour_detection': True,
            'min_object_size': 2
        },
        'executor': {
            'timeout_per_hypothesis': 5,
            'enable_parallel_execution': True
        },
        'verifier': {
            'strict_validation': True,
            'allow_approximate_matches': False
        }
    }
