import json
import pickle
import numpy as np
import os

class ModelConfig():
    def __init__(self):

        self.mnum = 1001
        self.rnum = 5
        self.rate_hidden = 5
        self.mhidden = 25
        self.his = 20
        self.hidden = self.mhidden + self.rate_hidden
        self.head_num = 3




