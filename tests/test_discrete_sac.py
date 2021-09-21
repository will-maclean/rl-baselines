from algorithm_test_framework import AlgorithmTestCase
from unittest import TestCase


class TestDiscreteSAC(AlgorithmTestCase, TestCase):

    def test_run_config(self):
        super().test_run_config("test_config/DiscreteSAC.yaml", "CartPole-v0")
