import sys
import pandas as pd
import numpy as np

trainingData = pd.read_excel(sys.argv[1])

# Classifications
# 1. Count the number of each classification
#data = np.array([], dtype =[('name', 'U10'), ('occurances', 'i10')])
weatherDescriptionCounts = pd.Series(trainingData['weather_descriptions'].value_counts())

weatherDescriptionProb = weatherDescriptionCounts.copy(deep=True)
weatherDescriptionProb = weatherDescriptionProb.apply(lambda x : x/(weatherDescriptionCounts.sum())) #p(Ck)



# Categorical Data
# 1. Run through series counting each attribute
#     a. Store attributes in a numpy array and count in another array
#     b. If an attribute is not in the array, concatenate to the end, otherwise increment associated ValueError
# 2. Calculate probabilites given each classification
# 3. Multiply

