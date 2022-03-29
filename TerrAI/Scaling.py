import numpy as np
import pandas as pd
import re

class Scaling():
    def __init__(self):
        self.scaling_data = pd.read_csv('../Dataset/Scaling.csv')
    
    def _simple_scale(self, x, min_x, max_x):
        # Scale values between min and max value, expected output min=0, max=1
        return (x - min_x) / (max_x - min_x)
    
    def scale(self, data):
        data = data.copy()
        columns = data.columns.values.tolist()
    
        for column in columns:
            raw_name = re.sub(r'\d+_', '', column) # The column name without the number part
            maxVal, minVal = self.scaling_data[raw_name] # Min max value for scaling
            
            # 1. Scale values according to scaling_data
            data[column] = self._simple_scale(data[column], minVal, maxVal)
        
        # 2. Check if any value is above 1 or bellow 0
        # Throw error when not fulfilled
        if (data.values < 0).any() or (data.values > 1).any():
            print()
            raise Exception('Scaling', f'{len(data.values < 0)} Values are found to be above 1 and {len(data.values > 1)} bellow 0')
        return data
