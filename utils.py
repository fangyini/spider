import numpy as np

class Paras:
    '''
    This class stores all the hyperparameters.
    '''
    def __init__(self):
        self.cell_size = 10000
        self.cycle = 6
        self.softsign_scale = 1
        self.device = None
        self.per = 100
        self.isKNN = True
        self.seed = 500
        self.KNN = 100
        self.predict_q_value_mode = False
        self.num_parallel = 1
        self.LAM4EVA = 0.5
        self.pre_actions = 21
        self.add_time_info = True
        self.skip_action = 1
        self.action_size = 50
