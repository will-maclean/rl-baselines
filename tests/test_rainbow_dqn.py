from algorithm_test_framework import AlgorithmTestCase
from unittest import TestCase


class TestRainbowDQN(AlgorithmTestCase, TestCase):

    def test_run_config(self):
        super().test_run_config("test_config/RainbowDQN.yaml", "CartPole-v0")
