#!/usr/bin/env python3
"""
ARC Prize 2025 - Basic Pipeline Testing Experiments

This script runs basic tests on the ARC reasoning pipeline
to validate functionality and collect performance metrics.
"""

import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path for imports
parent_path = Path(__file__).parent.parent
sys.path.insert(0, str(parent_path))

from src.data.loader import ARCDataLoader
from src.core.pipeline import ARCReasoningPipeline, PipelineConfig
from src.core.types import Task

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiments/logs/basic_pipeline_test.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def test_pipeline_configurations():
    """Test different pipeline configurations"""
    logger.info("Testing different pipeline configurations...")
    
    configurations = [
        ("symbolic_only", PipelineConfig(
            enable_llm=False,
            enable_vision=False, 
            enable_symbolic=True,
            max_hypotheses_per_component=5,
            max_total_hypotheses=10
        )),
        ("all_disabled", PipelineConfig(
            enable_llm=False,
            enable_vision=False,
            enable_symbolic=False
        )),
        ("max_hypotheses", PipelineConfig(
            enable_symbolic=True,
            max_hypotheses_per_component=10,
            max_total_hypotheses=20
        ))
    ]
    
    # Load test data
    loader = ARCDataLoader("data")
    training_tasks = loader.load_training_tasks()[:10]  # Test with first 10 tasks
    
    results = {}
    
    for config_name, config in configurations:
        logger.info(f"Testing configuration: {config_name}")
        pipeline = ARCReasoningPipeline(config)
        
        config_results = {
            'total_time': 0.0,
            'total_hypotheses': 0,
            'total_predictions': 0,
            'tasks_solved': 0,
            'task_times': []
        }
        
        for i, task in enumerate(training_tasks):
            logger.info(f"  Task {i+1}/{len(training_tasks)}: {task.task_id}")
            
            start_time = time.time()
            solution = pipeline.solve_task(task)
            task_time = time.time() - start_time
            
            config_results['total_time'] += task_time
            config_results['total_hypotheses'] += len(solution.hypotheses)
            config_results['total_predictions'] += len(solution.predictions)
            config_results['task_times'].append(task_time)
            
            if solution.predictions:
                config_results['tasks_solved'] += 1
        
        # Calculate averages
        config_results['avg_time_per_task'] = config_results['total_time'] / len(training_tasks)
        config_results['avg_hypotheses_per_task'] = config_results['total_hypotheses'] / len(training_tasks)
        config_results['success_rate'] = config_results['tasks_solved'] / len(training_tasks)
        
        results[config_name] = config_results
        
        logger.info(f"  Results: {config_results['success_rate']:.1%} success, "
                   f"{config_results['avg_time_per_task']:.3f}s avg time, "
                   f"{config_results['avg_hypotheses_per_task']:.1f} avg hypotheses")
    
    return results


def test_component_performance():
    """Test performance of individual components"""
    logger.info("Testing individual component performance...")
    
    # Load test data
    loader = ARCDataLoader("data")
    training_tasks = loader.load_training_tasks()[:5]
    
    # Test symbolic solver only
    config = PipelineConfig(
        enable_llm=False,
        enable_vision=False,
        enable_symbolic=True,
        max_hypotheses_per_component=10
    )
    
    pipeline = ARCReasoningPipeline(config)
    
    component_stats = {
        'preprocessor': {'total_time': 0.0, 'calls': 0},
        'symbolic_solver': {'total_time': 0.0, 'calls': 0, 'hypotheses': 0},
        'verifier': {'total_time': 0.0, 'calls': 0, 'verified': 0},
        'executor': {'total_time': 0.0, 'calls': 0, 'predictions': 0}
    }
    
    for task in training_tasks:
        logger.info(f"Testing components with task: {task.task_id}")
        
        # Detailed timing would need to be added to pipeline components
        # For now, just run the full pipeline
        solution = pipeline.solve_task(task)
        
        logger.info(f"  Hypotheses: {len(solution.hypotheses)}")
        logger.info(f"  Predictions: {len(solution.predictions)}")
        logger.info(f"  Time: {solution.execution_time:.3f}s")
    
    return component_stats


def test_error_handling():
    """Test pipeline error handling with malformed inputs"""
    logger.info("Testing error handling...")
    
    config = PipelineConfig(enable_symbolic=True)
    pipeline = ARCReasoningPipeline(config)
    
    # Test cases that might cause errors
    test_cases = [
        # Empty task
        Task(task_id="empty", train_pairs=[], test_inputs=[]),
        
        # Malformed grids (different sizes)
        Task(
            task_id="malformed",
            train_pairs=[{
                'input': [[1, 0], [0]],  # Irregular shape
                'output': [[1, 0], [0, 1]]
            }],
            test_inputs=[[[1, 0], [0, 1]]]
        ),
        
        # Very large grids
        Task(
            task_id="large",
            train_pairs=[{
                'input': [[0] * 50 for _ in range(50)],
                'output': [[1] * 50 for _ in range(50)]
            }],
            test_inputs=[[[0] * 50 for _ in range(50)]]
        )
    ]
    
    error_results = {}
    
    for test_case in test_cases:
        logger.info(f"Testing error case: {test_case.task_id}")
        
        try:
            solution = pipeline.solve_task(test_case)
            error_results[test_case.task_id] = {
                'status': 'success',
                'hypotheses': len(solution.hypotheses),
                'predictions': len(solution.predictions),
                'time': solution.execution_time
            }
            logger.info(f"  Handled successfully")
            
        except Exception as e:
            error_results[test_case.task_id] = {
                'status': 'error',
                'error': str(e),
                'error_type': type(e).__name__
            }
            logger.error(f"  Error: {e}")
    
    return error_results


def main():
    """Run all pipeline tests"""
    logger.info("Starting basic pipeline tests...")
    
    # Create logs directory
    logs_dir = Path("experiments/logs")
    logs_dir.mkdir(exist_ok=True)
    
    results = {}
    
    try:
        # Test 1: Configuration comparison
        logger.info("=== Test 1: Configuration Comparison ===")
        results['configurations'] = test_pipeline_configurations()
        
        # Test 2: Component performance  
        logger.info("\n=== Test 2: Component Performance ===")
        results['components'] = test_component_performance()
        
        # Test 3: Error handling
        logger.info("\n=== Test 3: Error Handling ===")
        results['errors'] = test_error_handling()
        
        # Save results
        import json
        results_file = logs_dir / f"pipeline_test_results_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_file}")
        
        # Print summary
        logger.info("\n=== Summary ===")
        if 'configurations' in results:
            for config_name, config_results in results['configurations'].items():
                logger.info(f"{config_name}: {config_results['success_rate']:.1%} success, "
                           f"{config_results['avg_time_per_task']:.3f}s avg")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    main()
