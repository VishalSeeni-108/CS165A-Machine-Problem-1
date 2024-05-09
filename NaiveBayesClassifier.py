import sys
import pandas as pd
import numpy as np

trainingData = pd.read_excel(sys.argv[1])
print(trainingData['temperature'])

# Classifications
# 1. Use same method as below to collect all classifications

# Categorical Data
# 1. Run through series counting each attribute
#     a. Store attributes in a numpy array and count in another array
#     b. If an attribute is not in the array, concatenate to the end, otherwise increment associated ValueError
# 2. Calculate probabilites given each classification
# 3. Multiply

