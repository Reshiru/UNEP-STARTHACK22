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
    
    def _columns_difference(self, data):
        main_cols = data.columns.tolist()
        main_cols.remove('Sample_ID')
        main_cols.remove('Label')
        return main_cols
    
    def reshape_and_scale_X(self, data):
        main_cols = self._columns_difference(data)
        
        # Scale X values
        data = self.scale(data[main_cols])
        
        # Group by features
        groups = sorted(set([re.sub(r'\d+_', '', column) for column in main_cols]))

        # Results in shape: (rows, features, values) -> expected 9 features with 5x5 (25) values
        grouped_inputs = []
        for group in groups:
            # Combine columns based on their name as a single flat feature: (rows, 25)
            grouped_input = data[[column for column in main_cols if column.endswith(group)]].to_numpy()
            # Remap to (rows, 5, 5)
            reshaped_input = np.reshape(grouped_input, (len(data), 5,5))
            # Transpose (5,5) part to match CELLID from Data dictionary.docx: (rows, 5, 5)
            transposed_input = np.transpose(reshaped_input, [0,2,1])
            data_group = np.expand_dims(transposed_input, axis=1)
            grouped_inputs.append(data_group)
        
        # Stack, expected: (rows, features, 5, 5)
        return np.hstack(grouped_inputs)