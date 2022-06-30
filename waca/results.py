from rough_imp import WACA
import os 

import numpy as np


AUTH = WACA()

directory = "waca\\user_data"

def fill_file(dir, csv):
    for root, subdirectories, files in os.walk(dir):
        for subdirectory in subdirectories:
            try:
                AUTH.label_vector(subdirectory, 1, csv)
            except:
                pass    

fill_file(directory, 'USER-FEATURE-DATA.csv')
  
