import sys
import pandas as pd
import numpy as np

trainingData = pd.read_excel(sys.argv[1])

def categoricalConditionalProb(category, descriptor):
        tempCounts = pd.Series(trainingData.loc[trainingData['weather_descriptions'].shift(periods=-1) == category, descriptor].value_counts()) #shift so that we are checking for FUTURE weather condition
        tempProbs = tempCounts.copy(deep = True)
        tempProbs = tempProbs.apply(lambda x : x / (tempCounts.sum()))
        # print("Category: " + category)
        # print(tempCounts)
        # print(tempProbs)
        # print('\n')
        return tempProbs

def conditionalProb(classification, humidity, cloudcover, precipitation): #pull conditional probabilities for given attributes and return probability for each classification
        classProb = weatherDescriptionProb[classification]
        humidityProb = humidityConditionalProbs[humidity][classification]
        cloudcoverProbs = cloudcoverConditionalProbs[cloudcover][classification]
        precipitationProbs = precipitationConditionalProbs[precipitation][classification]

        return classProb*(humidityProb * cloudcoverProbs * precipitationProbs)



# Classifications
# 1. Count the number of each classification (note that the classification is for FUTURE weather description)
weatherDescriptionCounts = pd.Series(trainingData['weather_descriptions'].value_counts())

weatherDescriptionProb = weatherDescriptionCounts.copy(deep=True)
weatherDescriptionProb = weatherDescriptionProb.apply(lambda x : x/(weatherDescriptionCounts.sum())) #p(Ck)

#print(weatherDescriptionCounts.index)



#Calculating feature conditional probabilities 
humidityConditionalProbs = ((weatherDescriptionCounts.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'humidity'))
cloudcoverConditionalProbs = ((weatherDescriptionCounts.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'cloudcover'))
precipitationConditionalProbs = ((weatherDescriptionCounts.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'precip'))
print(conditionalProb('Clear', 'Moderate humidity', 'Mostly clear', 'No precipitation'))

