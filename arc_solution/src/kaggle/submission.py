"""
ARC Prize 2025 - Kaggle Submission Management

This module handles the creation and validation of Kaggle submissions
for the ARC Prize 2025 competition.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime
import numpy as np

from ..core.types import Task, Solution
from ..data.loader import ARCDataLoader

logger = logging.getLogger(__name__)


class KaggleSubmissionManager:
    """
    Manages submission creation, validation, and formatting for Kaggle competition.
    
    Handles:
    - Prediction formatting according to Kaggle requirements
    - Submission file generation
    - Validation against sample submission format
    - Multiple attempt ranking and selection
    """
    
    def __init__(self, output_dir: str = "submissions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load sample submission format for validation
        self.sample_submission = None
        self._load_sample_submission()
    
    def _load_sample_submission(self) -> None:
        """Load sample submission format for validation"""
        try:
            sample_path = Path("data/sample_submission.json")
            if sample_path.exists():
                with open(sample_path, 'r') as f:
                    self.sample_submission = json.load(f)
                logger.info("Sample submission format loaded")
            else:
                logger.warning("Sample submission file not found")
        except Exception as e:
            logger.error(f"Error loading sample submission: {e}")
    
    def format_solution_for_submission(self, task_id: str, solution: Solution) -> Dict[str, List[List[List[int]]]]:
        """
        Format a solution according to Kaggle submission requirements.
        
        Args:
            task_id: Task identifier
            solution: Solution object with predictions
            
        Returns:
            Formatted submission entry for this task
        """
        # ARC submissions require up to 2 attempts per task
        attempts = []
        
        # Get best predictions from solution
        predictions = solution.predictions[:2]  # Take top 2 predictions
        
        for prediction in predictions:
            if hasattr(prediction, 'grid') and prediction.grid is not None:
                # Convert prediction grid to list format
                grid = prediction.grid
                if isinstance(grid, np.ndarray):
                    grid = grid.tolist()
                attempts.append(grid)
            elif hasattr(prediction, 'output') and prediction.output is not None:
                # Alternative: prediction has 'output' attribute
                grid = prediction.output
                if isinstance(grid, np.ndarray):
                    grid = grid.tolist()
                attempts.append(grid)
        
        # Ensure we have exactly 2 attempts (pad with empty if needed)
        while len(attempts) < 2:
            # Add empty grid as fallback
            attempts.append([[0]])
        
        # Limit to 2 attempts
        attempts = attempts[:2]
        
        return {task_id: attempts}
    
    def create_submission_file(self, 
                             solutions: Dict[str, Solution],
                             filename: Optional[str] = None,
                             description: str = "") -> Path:
        """
        Create a complete Kaggle submission file.
        
        Args:
            solutions: Dictionary mapping task_id -> Solution
            filename: Optional custom filename
            description: Description of this submission
            
        Returns:
            Path to the created submission file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"arc_submission_{timestamp}.json"
        
        submission_path = self.output_dir / filename
        submission_data = {}
        
        # Format each solution
        for task_id, solution in solutions.items():
            task_submission = self.format_solution_for_submission(task_id, solution)
            submission_data.update(task_submission)
        
        # Add metadata
        metadata = {
            "submission_metadata": {
                "timestamp": datetime.now().isoformat(),
                "description": description,
                "total_tasks": len(solutions),
                "submission_file": filename
            }
        }
        
        # Write submission file
        final_submission = {**submission_data}
        
        with open(submission_path, 'w') as f:
            json.dump(final_submission, f, indent=2)
        
        # Write metadata separately
        metadata_path = submission_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Submission file created: {submission_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        # Validate submission
        if self._validate_submission(submission_path):
            logger.info("✅ Submission validation passed")
        else:
            logger.warning("⚠️  Submission validation failed")
        
        return submission_path
    
    def _validate_submission(self, submission_path: Path) -> bool:
        """Validate submission format"""
        try:
            with open(submission_path, 'r') as f:
                submission = json.load(f)
            
            # Check if sample submission is available for format validation
            if self.sample_submission:
                return self._validate_against_sample(submission)
            else:
                return self._validate_basic_format(submission)
                
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False
    
    def _validate_against_sample(self, submission: Dict) -> bool:
        """Validate against sample submission format"""
        sample_tasks = set(self.sample_submission.keys())
        
        # Remove metadata if present
        submission_tasks = set(k for k in submission.keys() if not k.startswith('submission_'))
        
        # Check task coverage
        missing_tasks = sample_tasks - submission_tasks
        extra_tasks = submission_tasks - sample_tasks
        
        if missing_tasks:
            logger.warning(f"Missing tasks: {missing_tasks}")
        
        if extra_tasks:
            logger.info(f"Extra tasks (will be ignored): {extra_tasks}")
        
        # Validate format for each task
        valid_format = True
        for task_id in sample_tasks & submission_tasks:
            if not self._validate_task_format(task_id, submission[task_id]):
                valid_format = False
        
        return valid_format and len(missing_tasks) == 0
    
    def _validate_basic_format(self, submission: Dict) -> bool:
        """Basic format validation when sample is not available"""
        for task_id, attempts in submission.items():
            if task_id.startswith('submission_'):  # Skip metadata
                continue
                
            if not self._validate_task_format(task_id, attempts):
                return False
        
        return True
    
    def _validate_task_format(self, task_id: str, attempts: Any) -> bool:
        """Validate format for a single task"""
        try:
            # Should be a list of 1-2 attempts
            if not isinstance(attempts, list):
                logger.error(f"Task {task_id}: attempts should be a list")
                return False
            
            if len(attempts) == 0 or len(attempts) > 2:
                logger.error(f"Task {task_id}: should have 1-2 attempts, got {len(attempts)}")
                return False
            
            # Each attempt should be a 2D grid
            for i, attempt in enumerate(attempts):
                if not isinstance(attempt, list):
                    logger.error(f"Task {task_id}, attempt {i}: should be a 2D list")
                    return False
                
                # Check if it's a valid 2D grid
                if not all(isinstance(row, list) for row in attempt):
                    logger.error(f"Task {task_id}, attempt {i}: should be a 2D list")
                    return False
                
                # Check grid dimensions
                if len(attempt) == 0:
                    logger.error(f"Task {task_id}, attempt {i}: grid is empty")
                    return False
                
                row_length = len(attempt[0])
                if not all(len(row) == row_length for row in attempt):
                    logger.error(f"Task {task_id}, attempt {i}: irregular grid shape")
                    return False
                
                # Check values are integers 0-9
                for row in attempt:
                    for val in row:
                        if not isinstance(val, int) or val < 0 or val > 9:
                            logger.error(f"Task {task_id}, attempt {i}: invalid value {val}")
                            return False
            
            return True
            
        except Exception as e:
            logger.error(f"Validation error for task {task_id}: {e}")
            return False
    
    def create_test_submission(self) -> Path:
        """Create a test submission file using test dataset"""
        logger.info("Creating test submission...")
        
        # Load test tasks
        loader = ARCDataLoader("data")
        test_tasks = loader.load_test_tasks()
        
        # Create dummy solutions (for testing)
        solutions = {}
        for task in test_tasks:
            # Create dummy predictions based on first test input
            if task.test_inputs:
                test_input = task.test_inputs[0]
                
                # Simple fallback: copy input as prediction
                dummy_prediction = type('DummyPrediction', (), {
                    'grid': test_input,
                    'confidence': 0.1
                })()
                
                dummy_solution = Solution(
                    task_id=task.task_id,
                    predictions=[dummy_prediction],
                    hypotheses=[],
                    execution_time=0.01,
                    metadata={'method': 'dummy_fallback'}
                )
                
                solutions[task.task_id] = dummy_solution
        
        return self.create_submission_file(
            solutions, 
            filename="test_submission.json",
            description="Test submission with dummy predictions"
        )
    
    def analyze_submission(self, submission_path: Path) -> Dict[str, Any]:
        """Analyze a submission file and provide statistics"""
        with open(submission_path, 'r') as f:
            submission = json.load(f)
        
        # Remove metadata
        task_data = {k: v for k, v in submission.items() if not k.startswith('submission_')}
        
        analysis = {
            'total_tasks': len(task_data),
            'tasks_with_2_attempts': 0,
            'tasks_with_1_attempt': 0,
            'grid_sizes': [],
            'unique_colors_used': set(),
            'avg_grid_area': 0
        }
        
        total_area = 0
        
        for task_id, attempts in task_data.items():
            num_attempts = len(attempts)
            if num_attempts == 2:
                analysis['tasks_with_2_attempts'] += 1
            elif num_attempts == 1:
                analysis['tasks_with_1_attempt'] += 1
            
            # Analyze first attempt
            if attempts:
                grid = attempts[0]
                height = len(grid)
                width = len(grid[0]) if grid else 0
                area = height * width
                
                analysis['grid_sizes'].append((height, width))
                total_area += area
                
                # Collect colors
                for row in grid:
                    for val in row:
                        analysis['unique_colors_used'].add(val)
        
        analysis['avg_grid_area'] = total_area / len(task_data) if task_data else 0
        analysis['unique_colors_used'] = sorted(list(analysis['unique_colors_used']))
        
        return analysis

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

from ..core.types import Grid, TaskSolution
from ..data.loader import ARCDataLoader


class KaggleSubmissionGenerator:
    """
    Generates submissions in Kaggle ARC Prize format.
    
    Expected format:
    {
        "task_id_1": [
            [[0,1,2], [3,4,5]],  # attempt_1
            [[0,1,2], [3,4,5]]   # attempt_2  
        ],
        "task_id_2": [...]
    }
    """
    
    def __init__(self, output_dir: str = "kaggle_submission"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def generate_submission(self, 
                          solutions: List[TaskSolution], 
                          filename: str = None) -> str:
        """
        Generate Kaggle submission file from task solutions
        
        Args:
            solutions: List of TaskSolution objects
            filename: Custom filename (optional)
            
        Returns:
            Path to generated submission file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"submission_{timestamp}.json"
        
        submission_data = {}
        
        for solution in solutions:
            task_id = solution.task_id
            
            # Extract predictions for this task
            if solution.predictions and len(solution.predictions) > 0:
                task_predictions = []
                
                # For each test input, get attempt_1 and attempt_2
                for test_prediction in solution.predictions:
                    attempt_1 = test_prediction.get('attempt_1', [[0]])  # Fallback
                    attempt_2 = test_prediction.get('attempt_2', attempt_1)  # Use attempt_1 as fallback
                    
                    task_predictions.extend([attempt_1, attempt_2])
                
                submission_data[task_id] = task_predictions
            else:
                # No predictions available, use placeholder
                self.logger.warning(f"No predictions for task {task_id}, using placeholder")
                submission_data[task_id] = [[[0]], [[0]]]  # Placeholder for single test input
        
        # Save submission file
        submission_path = self.output_dir / filename
        with open(submission_path, 'w') as f:
            json.dump(submission_data, f, separators=(',', ':'))  # Compact format
        
        self.logger.info(f"Submission saved to {submission_path}")
        return str(submission_path)
    
    def validate_submission(self, submission_file: str, 
                          expected_tasks: List[str] = None) -> Dict[str, Any]:
        """
        Validate submission format and completeness
        
        Args:
            submission_file: Path to submission file
            expected_tasks: List of expected task IDs (optional)
            
        Returns:
            Validation results
        """
        try:
            with open(submission_file, 'r') as f:
                submission = json.load(f)
        except Exception as e:
            return {'valid': False, 'error': f'Failed to load submission: {e}'}
        
        validation_results = {
            'valid': True,
            'task_count': len(submission),
            'issues': []
        }
        
        # Check if expected tasks are present
        if expected_tasks:
            missing_tasks = set(expected_tasks) - set(submission.keys())
            extra_tasks = set(submission.keys()) - set(expected_tasks)
            
            if missing_tasks:
                validation_results['issues'].append(f"Missing tasks: {missing_tasks}")
            if extra_tasks:
                validation_results['issues'].append(f"Extra tasks: {extra_tasks}")
        
        # Validate each task's predictions
        for task_id, predictions in submission.items():
            if not isinstance(predictions, list):
                validation_results['issues'].append(f"Task {task_id}: predictions not a list")
                continue
            
            # Check each prediction
            for i, prediction in enumerate(predictions):
                if not self._is_valid_grid(prediction):
                    validation_results['issues'].append(
                        f"Task {task_id}, prediction {i}: invalid grid format"
                    )
        
        validation_results['valid'] = len(validation_results['issues']) == 0
        return validation_results
    
    def _is_valid_grid(self, grid: Any) -> bool:
        """Check if a grid has valid format"""
        try:
            if not isinstance(grid, list) or len(grid) == 0:
                return False
            
            if not all(isinstance(row, list) for row in grid):
                return False
            
            # Check consistent width
            width = len(grid[0])
            if not all(len(row) == width for row in grid):
                return False
            
            # Check valid cell values (0-9)
            for row in grid:
                for cell in row:
                    if not isinstance(cell, int) or cell < 0 or cell > 9:
                        return False
            
            return True
        except:
            return False
    
    def generate_test_submission(self, data_loader: ARCDataLoader, 
                               pipeline, filename: str = "test_submission.json") -> str:
        """
        Generate submission for test dataset using the pipeline
        
        Args:
            data_loader: ARC data loader
            pipeline: Reasoning pipeline
            filename: Output filename
            
        Returns:
            Path to generated submission file
        """
        self.logger.info("Generating test submission...")
        
        # Load test tasks
        test_tasks = data_loader.load_test_tasks()
        self.logger.info(f"Loaded {len(test_tasks)} test tasks")
        
        # Solve each test task
        solutions = []
        for i, task in enumerate(test_tasks):
            self.logger.info(f"Solving test task {i+1}/{len(test_tasks)}: {task.task_id}")
            solution = pipeline.solve_task(task)
            solutions.append(solution)
        
        # Generate submission
        submission_path = self.generate_submission(solutions, filename)
        
        # Validate submission
        expected_task_ids = [task.task_id for task in test_tasks]
        validation_results = self.validate_submission(submission_path, expected_task_ids)
        
        if validation_results['valid']:
            self.logger.info("✅ Submission generated and validated successfully")
        else:
            self.logger.warning(f"⚠️  Submission has issues: {validation_results['issues']}")
        
        return submission_path
    
    def create_sample_submission(self, data_loader: ARCDataLoader, 
                               filename: str = "sample_submission.json") -> str:
        """
        Create a sample submission with placeholder predictions
        
        Args:
            data_loader: ARC data loader  
            filename: Output filename
            
        Returns:
            Path to generated sample submission
        """
        test_tasks = data_loader.load_test_tasks()
        
        submission_data = {}
        for task in test_tasks:
            # Create placeholder predictions (copy first training input)
            if task.train_pairs:
                placeholder_grid = task.train_pairs[0]['input']
            else:
                placeholder_grid = [[0]]  # Minimal placeholder
            
            # Each test task may have multiple test inputs
            predictions = []
            for _ in task.test_inputs:
                predictions.extend([placeholder_grid, placeholder_grid])  # attempt_1, attempt_2
            
            submission_data[task.task_id] = predictions
        
        # Save sample submission
        submission_path = self.output_dir / filename
        with open(submission_path, 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        self.logger.info(f"Sample submission created: {submission_path}")
        return str(submission_path)
    
    def compare_submissions(self, submission1_path: str, 
                          submission2_path: str) -> Dict[str, Any]:
        """
        Compare two submissions and report differences
        
        Args:
            submission1_path: Path to first submission
            submission2_path: Path to second submission
            
        Returns:
            Comparison results
        """
        try:
            with open(submission1_path, 'r') as f:
                sub1 = json.load(f)
            with open(submission2_path, 'r') as f:
                sub2 = json.load(f)
        except Exception as e:
            return {'error': f'Failed to load submissions: {e}'}
        
        comparison = {
            'identical': True,
            'task_differences': [],
            'missing_in_sub1': [],
            'missing_in_sub2': [],
            'total_differences': 0
        }
        
        all_tasks = set(sub1.keys()) | set(sub2.keys())
        
        for task_id in all_tasks:
            if task_id not in sub1:
                comparison['missing_in_sub1'].append(task_id)
                comparison['identical'] = False
                continue
            
            if task_id not in sub2:
                comparison['missing_in_sub2'].append(task_id)
                comparison['identical'] = False
                continue
            
            if sub1[task_id] != sub2[task_id]:
                comparison['task_differences'].append(task_id)
                comparison['identical'] = False
        
        comparison['total_differences'] = (
            len(comparison['task_differences']) + 
            len(comparison['missing_in_sub1']) + 
            len(comparison['missing_in_sub2'])
        )
        
        return comparison
