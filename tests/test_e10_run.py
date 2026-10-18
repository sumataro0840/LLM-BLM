"""
E10's LoRA wrapper on a small synthetic layer: it starts equal to the frozen layer and its gradient reaches A and B.
"""
import importlib.util
import unittest
from pathlib import Path

import torch

_spec = importlib.util.spec_from_file_location(
    'e10_run', Path(__file__).resolve().parents[1] / 'experiments' / 'E10_chrono_lora' / 'run.py')
e10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(e10)


class TestLoRA(unittest.TestCase):
    def test_starts_equal_and_learns(self):
        torch.manual_seed(0)
        base = torch.nn.Linear(16, 8, bias=False)
        base.weight.requires_grad_(False)
        layer = e10.LoRA(base, rank=4, alpha=8)
        x = torch.randn(3, 16)
        torch.testing.assert_close(layer(x), base(x))          # B = 0 at the start
        layer(x).sum().backward()
        self.assertIsNotNone(layer.B.grad)
        self.assertGreater(float(layer.B.grad.abs().sum()), 0)
        self.assertIsNone(base.weight.grad)                     # the original weights stay frozen


if __name__ == '__main__':
    unittest.main()
