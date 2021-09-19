from .sac import SACAgent


class DiscreteSAC(SACAgent):
    def act(self, state, env, step=-1):
        pass

    def train_step(self, step):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def log(self, log_dict):
        pass

    def config(self):
        pass
