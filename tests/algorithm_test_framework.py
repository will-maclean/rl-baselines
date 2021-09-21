from abc import ABC, abstractmethod

from runner import RLRunner


class AlgorithmTestCase(ABC):

    @abstractmethod
    def test_run_config(self, config_file, env_name):
        """
        All algorithms should run this test so we can ensure they can load a config and run

        :param config_file: config file for agent in specified environment. Config should be for a VERY short test i.e. test
        should run for less than 30 seconds ideally.

        :param env_name: name of test environment
        """
        runner = RLRunner(env_name, config_file)
        runner.start()
