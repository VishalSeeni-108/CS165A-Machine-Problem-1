import sys
import pandas as pd
import numpy as np

trainingData = pd.read_excel(sys.argv[1])

def categoricalConditionalProb(category, descriptor):
        tempCounts = pd.Series(trainingData.loc[trainingData['weather_descriptions'] == category, descriptor].value_counts())

        tempProbs = tempCounts.copy(deep = True)
        tempProbs = tempProbs.apply(lambda x : x / (tempCounts.sum()))
        print("Category: " + category)
        print(tempCounts)
        print(tempProbs)
        print('\n')

# Classifications
# 1. Count the number of each classification
weatherDescriptionCounts = pd.Series(trainingData['weather_descriptions'].value_counts())

weatherDescriptionProb = weatherDescriptionCounts.copy(deep=True)
weatherDescriptionProb = weatherDescriptionProb.apply(lambda x : x/(weatherDescriptionCounts.sum())) #p(Ck)

print(weatherDescriptionCounts.index)

# Categorical Data
# First take subset from each classification
# Count number of each attribute

#humidity 
print(pd.Series(trainingData['weather_descriptions'].unique()))
categoricalConditionalProb('Clear', 'humidity')
categoricalConditionalProb('Sunny', 'humidity')



