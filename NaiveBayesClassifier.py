import sys
import pandas as pd
import numpy as np

trainingData = pd.read_excel(sys.argv[1])

def categoricalConditionalProb(category, descriptor):
        tempCounts = pd.Series(trainingData.loc[trainingData['weather_descriptions'].shift(periods=-1) == category, descriptor].value_counts()) #shift so that we are checking for FUTURE weather condition
        tempProbs = tempCounts.copy(deep = True)
        tempProbs = tempProbs.apply(lambda x : x / (tempCounts.sum()))
        print("Category: " + category)
        print(tempCounts)
        print(tempProbs)
        print('\n')
        return tempProbs



# Classifications
# 1. Count the number of each classification (note that the classification is for FUTURE weather description)
weatherDescriptionCounts = pd.Series(trainingData['weather_descriptions'].value_counts())

weatherDescriptionProb = weatherDescriptionCounts.copy(deep=True)
weatherDescriptionProb = weatherDescriptionProb.apply(lambda x : x/(weatherDescriptionCounts.sum())) #p(Ck)

print(weatherDescriptionCounts.index)



#humidity 
print(pd.Series(trainingData['weather_descriptions'].unique()))
print(categoricalConditionalProb('Clear', 'humidity'))
print(categoricalConditionalProb('Sunny', 'humidity'))
print(categoricalConditionalProb('Light freezing rain', 'precip'))

#for a given descriptor, we need to apply categoricalConditiaonalProb with all categories, then store it in an associated descriptor datafram
humidityConditionalProbs = pd.DataFrame()
humidityConditionalProbs.concat(weatherDescriptionCounts.index.apply(lambda x : categoricalConditionalProb(x, 'humidity')))


