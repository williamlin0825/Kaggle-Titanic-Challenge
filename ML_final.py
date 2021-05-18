# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib

train=pd.read_csv(r'C:\Users\admin\Desktop\train.csv')
test=pd.read_csv(r'C:\Users\admin\Desktop\test.csv')

Y_label = train.Survived
train.drop('Survived', 1, inplace=True)
train.drop('Cabin', 1, inplace=True)
print(train)
