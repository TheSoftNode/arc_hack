"""
ARC Prize 2025 - Main Reasoning Pipeline

This module orchestrates the multi-agent neuro-symbolic reasoning system
for solving ARC tasks through hypothesis generation, execution, and verification.
"""

import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .types import Task, Hypothesis, TaskSolution
from .preprocessor import MultiModalPreprocessor
from ..hypothesis_generators.symbolic_solver import SymbolicSolver
from ..dsl.primitives import DSLPrimitives
from ..dsl.program_synthesizer import ProgramSynthesizer
from ..llm_reasoning.llm_reasoner import LLMReasoner
from ..vision.vision_solver import VisionSolver
from ..execution.executor import ExecutionEngine
from ..verification.verifier import HypothesisVerifier


@dataclass
class PipelineConfig:
    """Configuration for the reasoning pipeline"""
    max_hypotheses_per_component: int = 10
    max_total_hypotheses: int = 50
    verification_threshold: float = 0.5
    execution_timeout: float = 30.0
    enable_llm: bool = True
    enable_vision: bool = True
    enable_symbolic: bool = True
    llm_provider: str = "openai"
    llm_model: str = "gpt-4"


class ARCReasoningPipeline:
    """
    Main pipeline that orchestrates the multi-agent reasoning system.
    
    Components:
    1. Preprocessor: Multi-modal grid analysis
    2. Symbolic Solver: Rule-based reasoning and program synthesis
    3. LLM Reasoner: Natural language reasoning and guidance
    4. Vision Solver: Computer vision pattern recognition
    5. Execution Engine: Safe program execution and validation
    6. Verifier: Hypothesis ranking and selection
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.preprocessor = MultiModalPreprocessor(config={})
        self.primitives = DSLPrimitives()
        self.program_synthesizer = ProgramSynthesizer(self.primitives)
        self.symbolic_solver = SymbolicSolver({})  # Pass empty config for now
        self.executor = ExecutionEngine(self.primitives)
        self.verifier = HypothesisVerifier(self.executor)
        
        # Optional components (depend on external libraries)
        self.llm_reasoner = None
        self.vision_solver = None
        
        if self.config.enable_llm:
            try:
                self.llm_reasoner = LLMReasoner(
                    provider=self.config.llm_provider,
                    model=self.config.llm_model
                )
            except Exception as e:
                self.logger.warning(f"LLM reasoner initialization failed: {e}")
        
        if self.config.enable_vision:
            try:
                self.vision_solver = VisionSolver()
            except Exception as e:
                self.logger.warning(f"Vision solver initialization failed: {e}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the pipeline"""
        logger = logging.getLogger('arc_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def solve_task(self, task: Task) -> TaskSolution:
        """
        Solve a single ARC task using the multi-agent reasoning pipeline
        
        Args:
            task: The ARC task to solve
            
        Returns:
            Complete task solution with ranked hypotheses and predictions
        """
        start_time = time.time()
        self.logger.info(f"Starting to solve task {task.task_id}")
        
        try:
            # Phase 1: Preprocessing
            self.logger.info("Phase 1: Preprocessing grids...")
            preprocessed_scenes = self._preprocess_task(task)
            
            # Phase 2: Hypothesis Generation
            self.logger.info("Phase 2: Generating hypotheses...")
            all_hypotheses = self._generate_hypotheses(task, preprocessed_scenes)
            self.logger.info(f"Generated {len(all_hypotheses)} hypotheses")
            
            # Phase 3: Hypothesis Verification and Ranking
            self.logger.info("Phase 3: Verifying and ranking hypotheses...")
            verification_results = self.verifier.verify_hypotheses(all_hypotheses, task)
            
            # Phase 4: Select Best Hypotheses
            best_hypotheses = self.verifier.select_best_hypotheses(verification_results)
            self.logger.info(f"Selected {len(best_hypotheses)} best hypotheses")
            
            # Phase 5: Generate Predictions
            self.logger.info("Phase 5: Generating predictions...")
            predictions = self._generate_predictions(best_hypotheses, task)
            
            # Compute confidence scores
            confidence_scores = [
                result.score for result in verification_results[:len(best_hypotheses)]
            ]
            
            execution_time = time.time() - start_time
            self.logger.info(f"Task solved in {execution_time:.2f} seconds")
            
            return TaskSolution(
                task_id=task.task_id,
                hypotheses=best_hypotheses,
                predictions=predictions,
                confidence_scores=confidence_scores,
                execution_time=execution_time,
                metadata={
                    'total_hypotheses_generated': len(all_hypotheses),
                    'verification_summary': self.verifier.get_verification_summary(verification_results),
                    'config': self.config
                }
            )
        
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error solving task {task.task_id}: {e}")
            
            return TaskSolution(
                task_id=task.task_id,
                hypotheses=[],
                predictions=[],
                confidence_scores=[],
                execution_time=execution_time,
                metadata={'error': str(e)}
            )
    
    def _preprocess_task(self, task: Task) -> List[Any]:
        """Preprocess all grids in the task"""
        scenes = []
        
        # Preprocess training examples
        for pair in task.train_pairs:
            input_scene = self.preprocessor.process_grid(pair['input'])
            output_scene = self.preprocessor.process_grid(pair['output'])
            scenes.extend([input_scene, output_scene])
        
        # Preprocess test inputs
        for test_input in task.test_inputs:
            test_scene = self.preprocessor.process_grid(test_input)
            scenes.append(test_scene)
        
        return scenes
    
    def _generate_hypotheses(self, task: Task, preprocessed_scenes: List[Any]) -> List[Hypothesis]:
        """Generate hypotheses using all available reasoning components"""
        all_hypotheses = []
        
        # Symbolic reasoning hypotheses
        if self.config.enable_symbolic:
            try:
                symbolic_hypotheses = self.symbolic_solver.generate_hypotheses(
                    task, preprocessed_scenes
                )
                all_hypotheses.extend(symbolic_hypotheses)
                self.logger.info(f"Symbolic solver generated {len(symbolic_hypotheses)} hypotheses")
            except Exception as e:
                self.logger.warning(f"Symbolic solver failed: {e}")
        
        # LLM reasoning hypotheses
        if self.llm_reasoner:
            try:
                llm_hypotheses = self.llm_reasoner.generate_hypotheses(
                    task, max_hypotheses=self.config.max_hypotheses_per_component
                )
                all_hypotheses.extend(llm_hypotheses)
                self.logger.info(f"LLM reasoner generated {len(llm_hypotheses)} hypotheses")
            except Exception as e:
                self.logger.warning(f"LLM reasoner failed: {e}")
        
        # Vision-based hypotheses
        if self.vision_solver:
            try:
                vision_hypotheses = self.vision_solver.generate_hypotheses(
                    task, max_hypotheses=self.config.max_hypotheses_per_component
                )
                all_hypotheses.extend(vision_hypotheses)
                self.logger.info(f"Vision solver generated {len(vision_hypotheses)} hypotheses")
            except Exception as e:
                self.logger.warning(f"Vision solver failed: {e}")
        
        # Limit total hypotheses
        if len(all_hypotheses) > self.config.max_total_hypotheses:
            # Keep hypotheses with highest initial confidence
            all_hypotheses.sort(key=lambda h: h.confidence, reverse=True)
            all_hypotheses = all_hypotheses[:self.config.max_total_hypotheses]
            self.logger.info(f"Limited to {len(all_hypotheses)} hypotheses")
        
        return all_hypotheses
    
    def _generate_predictions(self, hypotheses: List[Hypothesis], task: Task) -> List[Dict[str, Any]]:
        """Generate predictions for test inputs using verified hypotheses"""
        predictions = []
        
        for i, test_input in enumerate(task.test_inputs):
            test_predictions = {}
            
            # Generate predictions using top hypotheses
            for attempt_idx, hypothesis in enumerate(hypotheses[:2]):  # Top 2 attempts
                try:
                    if hypothesis.program:
                        execution_result = self.executor.execute_program(hypothesis.program, test_input)
                        if execution_result.success and execution_result.output:
                            test_predictions[f'attempt_{attempt_idx + 1}'] = execution_result.output
                        else:
                            # Fallback: return input grid if execution fails
                            test_predictions[f'attempt_{attempt_idx + 1}'] = test_input
                    else:
                        # No program available, return input grid
                        test_predictions[f'attempt_{attempt_idx + 1}'] = test_input
                
                except Exception as e:
                    self.logger.warning(f"Prediction generation failed for test {i}, attempt {attempt_idx + 1}: {e}")
                    test_predictions[f'attempt_{attempt_idx + 1}'] = test_input
            
            # Ensure we have both attempts
            if 'attempt_1' not in test_predictions:
                test_predictions['attempt_1'] = test_input
            if 'attempt_2' not in test_predictions:
                test_predictions['attempt_2'] = test_predictions.get('attempt_1', test_input)
            
            predictions.append(test_predictions)
        
        return predictions
    
    def batch_solve(self, tasks: List[Task]) -> List[TaskSolution]:
        """Solve multiple tasks in batch"""
        solutions = []
        
        for i, task in enumerate(tasks):
            self.logger.info(f"Solving task {i+1}/{len(tasks)}: {task.task_id}")
            solution = self.solve_task(task)
            solutions.append(solution)
        
        return solutions
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline components"""
        stats = {
            'components': {
                'preprocessor': True,
                'symbolic_solver': self.config.enable_symbolic,
                'llm_reasoner': self.llm_reasoner is not None,
                'vision_solver': self.vision_solver is not None,
                'executor': True,
                'verifier': True
            },
            'config': {
                'max_hypotheses_per_component': self.config.max_hypotheses_per_component,
                'max_total_hypotheses': self.config.max_total_hypotheses,
                'execution_timeout': self.config.execution_timeout,
                'llm_provider': self.config.llm_provider if self.llm_reasoner else None,
                'llm_model': self.config.llm_model if self.llm_reasoner else None
            }
        }
        
        return stats


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
