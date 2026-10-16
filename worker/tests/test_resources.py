"""Resource estimates must reject over-budget allocations, never invent GPUs."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.resources import plan


class ResourcePlanTests(unittest.TestCase):
    def test_small_data_uses_one_cpu_worker(self):
        self.assertEqual(plan('fleet_distributed_random_forest', 150, 4)['workers'], 1)

    def test_replicated_data_cannot_exceed_memory_budget(self):
        with self.assertRaises(ValueError):
            plan('fleet_distributed_random_forest', 50000, 784, memory_budget=1000)

    def test_one_accelerator_remains_one_worker(self):
        with patch('training.accelerated.available_devices', return_value=['cpu', 'mps']):
            result=plan('fleet_accelerated_logistic', 50000, 784)
        self.assertEqual((result['workers'], result['device']), (1, 'mps'))
