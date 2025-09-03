"""
ARC Prize 2025 - Experiment Configuration

Configuration and utilities for running experiments.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

from ..core.pipeline import PipelineConfig


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run"""
    experiment_name: str
    description: str
    pipeline_config: PipelineConfig
    dataset: str = "training"  # training, evaluation, test
    max_tasks: int = 10  # Maximum number of tasks to run
    save_results: bool = True
    save_predictions: bool = True
    log_level: str = "INFO"
    
    # Directories
    output_dir: str = "experiments/outputs"
    log_dir: str = "experiments/logs"


class ExperimentRunner:
    """Runs and manages experiments"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = f"{config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup directories
        self.output_dir = Path(config.output_dir) / self.experiment_id
        self.log_dir = Path(config.log_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Save experiment config
        self._save_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup experiment logging"""
        logger = logging.getLogger(f'experiment_{self.experiment_id}')
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # File handler
        log_file = self.log_dir / f"{self.experiment_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _save_config(self):
        """Save experiment configuration"""
        config_file = self.output_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(asdict(self.config), f, indent=2, default=str)
    
    def run_experiment(self, pipeline, data_loader):
        """Run the experiment"""
        self.logger.info(f"Starting experiment: {self.experiment_id}")
        start_time = time.time()
        
        try:
            # Load tasks
            if self.config.dataset == "training":
                tasks = data_loader.load_training_tasks()
            elif self.config.dataset == "evaluation":
                tasks = data_loader.load_evaluation_tasks()
            elif self.config.dataset == "test":
                tasks = data_loader.load_test_tasks()
            else:
                raise ValueError(f"Unknown dataset: {self.config.dataset}")
            
            # Limit number of tasks
            tasks = tasks[:self.config.max_tasks]
            self.logger.info(f"Running on {len(tasks)} tasks from {self.config.dataset} set")
            
            # Run pipeline on each task
            results = []
            for i, task in enumerate(tasks):
                self.logger.info(f"Processing task {i+1}/{len(tasks)}: {task.task_id}")
                
                task_start_time = time.time()
                solution = pipeline.solve_task(task)
                task_time = time.time() - task_start_time
                
                # Collect results
                result = {
                    'task_id': task.task_id,
                    'execution_time': task_time,
                    'num_hypotheses': len(solution.hypotheses),
                    'confidence_scores': solution.confidence_scores,
                    'success': len(solution.hypotheses) > 0,
                    'metadata': solution.metadata
                }
                
                if self.config.save_predictions:
                    result['predictions'] = solution.predictions
                
                results.append(result)
                self.logger.info(f"Task {task.task_id} completed in {task_time:.2f}s")
            
            # Save results
            if self.config.save_results:
                self._save_results(results)
            
            # Compute summary statistics
            summary = self._compute_summary(results)
            self._save_summary(summary)
            
            total_time = time.time() - start_time
            self.logger.info(f"Experiment completed in {total_time:.2f}s")
            
            return results, summary
        
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise
    
    def _save_results(self, results: List[Dict[str, Any]]):
        """Save detailed results"""
        results_file = self.output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def _save_summary(self, summary: Dict[str, Any]):
        """Save experiment summary"""
        summary_file = self.output_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary saved to {summary_file}")
    
    def _compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute experiment summary statistics"""
        if not results:
            return {}
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r['success'])
        total_time = sum(r['execution_time'] for r in results)
        avg_time = total_time / total_tasks
        
        # Hypothesis statistics
        total_hypotheses = sum(r['num_hypotheses'] for r in results)
        avg_hypotheses = total_hypotheses / total_tasks
        
        # Confidence statistics
        all_confidences = []
        for result in results:
            all_confidences.extend(result.get('confidence_scores', []))
        
        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        max_confidence = max(all_confidences) if all_confidences else 0
        
        summary = {
            'experiment_id': self.experiment_id,
            'total_tasks': total_tasks,
            'successful_tasks': successful_tasks,
            'success_rate': successful_tasks / total_tasks,
            'total_execution_time': total_time,
            'average_task_time': avg_time,
            'total_hypotheses_generated': total_hypotheses,
            'average_hypotheses_per_task': avg_hypotheses,
            'average_confidence': avg_confidence,
            'max_confidence': max_confidence,
            'tasks_with_high_confidence': sum(1 for r in results 
                                            if r.get('confidence_scores') and max(r['confidence_scores']) > 0.8)
        }
        
        return summary
