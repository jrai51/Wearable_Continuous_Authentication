from rough_imp import WACA
import os 

AUTH = WACA()

directory = "waca\\user_data"
    
for root, subdirectories, files in os.walk(directory):
    for subdirectory in subdirectories:
        try:
            AUTH.label_vector(subdirectory, 1)
        except:
            pass    
  
