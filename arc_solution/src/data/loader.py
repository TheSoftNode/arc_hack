"""
ARC Prize 2025 - Data Loader

This module provides utilities for loading and parsing ARC dataset files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from ..core.types import Task


class ARCDataLoader:
    """
    Loads and parses ARC dataset files into our internal Task format.
    
    Handles:
    - Training challenges and solutions
    - Evaluation challenges and solutions  
    - Test challenges
    - Sample submission format
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        if data_dir is None:
            # Default to data directory relative to this file
            self.data_dir = Path(__file__).parent.parent.parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        self.logger = logging.getLogger(__name__)
        
        # Dataset file paths
        self.files = {
            'training_challenges': self.data_dir / 'arc-agi_training_challenges.json',
            'training_solutions': self.data_dir / 'arc-agi_training_solutions.json',
            'evaluation_challenges': self.data_dir / 'arc-agi_evaluation_challenges.json',
            'evaluation_solutions': self.data_dir / 'arc-agi_evaluation_solutions.json',
            'test_challenges': self.data_dir / 'arc-agi_test_challenges.json',
            'sample_submission': self.data_dir / 'sample_submission.json'
        }
    
    def load_training_tasks(self) -> List[Task]:
        """Load all training tasks with solutions"""
        return self._load_tasks_with_solutions(
            self.files['training_challenges'],
            self.files['training_solutions']
        )
    
    def load_evaluation_tasks(self) -> List[Task]:
        """Load all evaluation tasks with solutions"""
        return self._load_tasks_with_solutions(
            self.files['evaluation_challenges'],
            self.files['evaluation_solutions']
        )
    
    def load_test_tasks(self) -> List[Task]:
        """Load test tasks (no solutions provided)"""
        return self._load_tasks_without_solutions(self.files['test_challenges'])
    
    def load_sample_submission(self) -> Dict[str, Any]:
        """Load sample submission format"""
        try:
            with open(self.files['sample_submission'], 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading sample submission: {e}")
            return {}
    
    def _load_tasks_with_solutions(self, challenges_file: Path, solutions_file: Path) -> List[Task]:
        """Load tasks that have both challenges and solutions"""
        try:
            # Load challenges
            with open(challenges_file, 'r') as f:
                challenges = json.load(f)
            
            # Load solutions
            with open(solutions_file, 'r') as f:
                solutions = json.load(f)
            
            tasks = []
            for task_id, challenge_data in challenges.items():
                if task_id in solutions:
                    solution_data = solutions[task_id]
                    task = self._create_task_with_solution(task_id, challenge_data, solution_data)
                    tasks.append(task)
                else:
                    self.logger.warning(f"No solution found for task {task_id}")
            
            self.logger.info(f"Loaded {len(tasks)} tasks with solutions")
            return tasks
        
        except Exception as e:
            self.logger.error(f"Error loading tasks with solutions: {e}")
            return []
    
    def _load_tasks_without_solutions(self, challenges_file: Path) -> List[Task]:
        """Load tasks that only have challenges (test set)"""
        try:
            with open(challenges_file, 'r') as f:
                challenges = json.load(f)
            
            tasks = []
            for task_id, challenge_data in challenges.items():
                task = self._create_task_without_solution(task_id, challenge_data)
                tasks.append(task)
            
            self.logger.info(f"Loaded {len(tasks)} test tasks")
            return tasks
        
        except Exception as e:
            self.logger.error(f"Error loading test tasks: {e}")
            return []
    
    def _create_task_with_solution(self, task_id: str, challenge_data: Dict, solution_data: List) -> Task:
        """Create Task object with known solutions"""
        # Extract training pairs
        train_pairs = []
        for train_example in challenge_data['train']:
            train_pairs.append({
                'input': train_example['input'],
                'output': train_example['output']
            })
        
        # Extract test inputs
        test_inputs = [test_example['input'] for test_example in challenge_data['test']]
        
        # Extract test outputs from solutions
        test_outputs = solution_data  # Solutions are provided as list of grids
        
        return Task(
            task_id=task_id,
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=test_outputs
        )
    
    def _create_task_without_solution(self, task_id: str, challenge_data: Dict) -> Task:
        """Create Task object without known solutions (for test set)"""
        # Extract training pairs
        train_pairs = []
        for train_example in challenge_data['train']:
            train_pairs.append({
                'input': train_example['input'],
                'output': train_example['output']
            })
        
        # Extract test inputs
        test_inputs = [test_example['input'] for test_example in challenge_data['test']]
        
        return Task(
            task_id=task_id,
            train_pairs=train_pairs,
            test_inputs=test_inputs,
            test_outputs=None  # No solutions for test set
        )
    
    def get_task_by_id(self, task_id: str, dataset: str = 'training') -> Optional[Task]:
        """Get a specific task by ID from the specified dataset"""
        if dataset == 'training':
            tasks = self.load_training_tasks()
        elif dataset == 'evaluation':
            tasks = self.load_evaluation_tasks()
        elif dataset == 'test':
            tasks = self.load_test_tasks()
        else:
            self.logger.error(f"Unknown dataset: {dataset}")
            return None
        
        for task in tasks:
            if task.task_id == task_id:
                return task
        
        return None
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the datasets"""
        try:
            training_tasks = self.load_training_tasks()
            evaluation_tasks = self.load_evaluation_tasks()
            test_tasks = self.load_test_tasks()
            
            stats = {
                'training': {
                    'count': len(training_tasks),
                    'avg_train_examples': sum(len(t.train_pairs) for t in training_tasks) / len(training_tasks) if training_tasks else 0,
                    'avg_test_inputs': sum(len(t.test_inputs) for t in training_tasks) / len(training_tasks) if training_tasks else 0
                },
                'evaluation': {
                    'count': len(evaluation_tasks),
                    'avg_train_examples': sum(len(t.train_pairs) for t in evaluation_tasks) / len(evaluation_tasks) if evaluation_tasks else 0,
                    'avg_test_inputs': sum(len(t.test_inputs) for t in evaluation_tasks) / len(evaluation_tasks) if evaluation_tasks else 0
                },
                'test': {
                    'count': len(test_tasks),
                    'avg_train_examples': sum(len(t.train_pairs) for t in test_tasks) / len(test_tasks) if test_tasks else 0,
                    'avg_test_inputs': sum(len(t.test_inputs) for t in test_tasks) / len(test_tasks) if test_tasks else 0
                }
            }
            
            return stats
        
        except Exception as e:
            self.logger.error(f"Error computing dataset stats: {e}")
            return {}
    
    def validate_dataset_files(self) -> Dict[str, bool]:
        """Validate that all required dataset files exist and are readable"""
        validation_results = {}
        
        for file_type, file_path in self.files.items():
            try:
                if file_path.exists() and file_path.is_file():
                    # Try to load and parse the JSON
                    with open(file_path, 'r') as f:
                        json.load(f)
                    validation_results[file_type] = True
                else:
                    validation_results[file_type] = False
            except Exception as e:
                self.logger.error(f"Error validating {file_type}: {e}")
                validation_results[file_type] = False
        
        return validation_results
