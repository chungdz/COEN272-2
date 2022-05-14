import json
import pickle
import numpy as np
import os

class ModelConfig():
    def __init__(self):

        self.mnum = 1001
        self.rnum = 6
        self.rate_hidden = 10
        self.mhidden = 50
        self.his = 20
        self.hidden = self.mhidden + self.rate_hidden
        self.head_num = 3




