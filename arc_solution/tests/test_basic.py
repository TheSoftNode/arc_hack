"""
ARC Prize 2025 - Basic Tests

Basic unit tests for core components of the ARC solution.
"""

import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.types import Task, Hypothesis
from core.pipeline import ARCReasoningPipeline, PipelineConfig
from data.loader import ARCDataLoader


class TestARCDataLoader(unittest.TestCase):
    """Test the ARC data loader functionality"""
    
    def setUp(self):
        self.loader = ARCDataLoader()
    
    def test_validate_dataset_files(self):
        """Test that dataset files exist and are valid"""
        validation_results = self.loader.validate_dataset_files()
        
        # Check that all files are valid
        for file_type, is_valid in validation_results.items():
            with self.subTest(file_type=file_type):
                self.assertTrue(is_valid, f"{file_type} file is not valid")
    
    def test_load_training_tasks(self):
        """Test loading training tasks"""
        tasks = self.loader.load_training_tasks()
        
        self.assertIsInstance(tasks, list)
        self.assertGreater(len(tasks), 0, "Should load at least one training task")
        
        # Check first task structure
        if tasks:
            task = tasks[0]
            self.assertIsInstance(task, Task)
            self.assertIsInstance(task.task_id, str)
            self.assertIsInstance(task.train_pairs, list)
            self.assertIsInstance(task.test_inputs, list)
    
    def test_get_dataset_stats(self):
        """Test dataset statistics computation"""
        stats = self.loader.get_dataset_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('training', stats)
        self.assertIn('evaluation', stats)
        self.assertIn('test', stats)


class TestPipelineBasics(unittest.TestCase):
    """Test basic pipeline functionality"""
    
    def setUp(self):
        self.config = PipelineConfig(
            max_hypotheses_per_component=3,
            max_total_hypotheses=5,
            enable_llm=False,  # Disable for testing
            enable_vision=False,  # Disable for testing
            enable_symbolic=True
        )
        self.pipeline = ARCReasoningPipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test that pipeline initializes correctly"""
        self.assertIsNotNone(self.pipeline.preprocessor)
        self.assertIsNotNone(self.pipeline.primitives)
        self.assertIsNotNone(self.pipeline.executor)
        self.assertIsNotNone(self.pipeline.verifier)
    
    def test_get_pipeline_stats(self):
        """Test pipeline statistics"""
        stats = self.pipeline.get_pipeline_stats()
        
        self.assertIsInstance(stats, dict)
        self.assertIn('components', stats)
        self.assertIn('config', stats)
    
    def test_simple_task_solving(self):
        """Test solving a simple identity task"""
        # Create a simple identity transformation task
        task = Task(
            task_id="test_identity",
            train_pairs=[
                {"input": [[1, 0], [0, 1]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 1], [1, 2]], "output": [[2, 1], [1, 2]]}
            ],
            test_inputs=[[[3, 0], [0, 3]]]
        )
        
        # Solve the task
        solution = self.pipeline.solve_task(task)
        
        # Check solution structure
        self.assertEqual(solution.task_id, "test_identity")
        self.assertIsInstance(solution.hypotheses, list)
        self.assertIsInstance(solution.predictions, list)
        self.assertIsInstance(solution.confidence_scores, list)
        self.assertGreater(solution.execution_time, 0)


class TestCoreTypes(unittest.TestCase):
    """Test core data types"""
    
    def test_task_creation(self):
        """Test Task object creation"""
        task = Task(
            task_id="test_001",
            train_pairs=[{"input": [[1]], "output": [[2]]}],
            test_inputs=[[[3]]]
        )
        
        self.assertEqual(task.task_id, "test_001")
        self.assertEqual(len(task.train_pairs), 1)
        self.assertEqual(len(task.test_inputs), 1)
    
    def test_hypothesis_creation(self):
        """Test Hypothesis object creation"""
        hypothesis = Hypothesis(
            transformations=[],
            confidence=0.8,
            description="Test hypothesis",
            generated_by="test"
        )
        
        self.assertEqual(hypothesis.confidence, 0.8)
        self.assertEqual(hypothesis.description, "Test hypothesis")
        self.assertEqual(hypothesis.generated_by, "test")


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    unittest.main(verbosity=2)
