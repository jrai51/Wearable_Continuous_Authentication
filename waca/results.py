from rough_imp import WACA
import os 

import numpy as np


AUTH = WACA()

directory = "waca\\user_data"

def fill_file(dir, test, csv):
    for root, subdirectories, files in os.walk(dir):
        for subdirectory in subdirectories:
            try:
                AUTH.label_vector(subdirectory, test, csv)
            except:
                pass    

fill_file(directory, 2, 'TEST2-FEATURE-DATA.csv')
  
