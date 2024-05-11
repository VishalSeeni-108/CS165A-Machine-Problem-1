import sys
import pandas as pd
import numpy as np
import pathlib as pl
import os
import re
import math

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

def numericalConditionalProb(category, descriptor, value): 
        categoryData = pd.Series(trainingData.loc[trainingData['weather_descriptions'].shift(periods=-1) == category, descriptor]) #Pulls temperatures for which next day is given category
        return (1 / (math.sqrt(2*math.pi*(math.pow(categoryData.std(), 2))))*(math.pow(math.e, -((math.pow(value - categoryData.mean(), 2))/(2*(math.pow(categoryData.std(), 2)))))))        

def conditionalProb(classification, humidity, cloudcover, precipitation): #pull conditional probabilities for given attributes and return probability for each classification
        classProb = weatherDescriptionProb[classification]
        humidityProb = humidityConditionalProbs[humidity][classification]
        cloudcoverProbs = cloudcoverConditionalProbs[cloudcover][classification]
        precipitationProbs = precipitationConditionalProbs[precipitation][classification]

        return classProb*(humidityProb * cloudcoverProbs * precipitationProbs)

def classificationProb(humidity, cloudcover, precipitation):
        return (weatherDescriptionCounts.index).to_series().apply(lambda x : conditionalProb(x, humidity, cloudcover, precipitation))
                                                                   
#Calculate classification probabilities
weatherDescriptionCounts = pd.Series(trainingData['weather_descriptions'].value_counts())
weatherDescriptionProb = weatherDescriptionCounts.copy(deep=True)
weatherDescriptionProb = weatherDescriptionProb.apply(lambda x : x/(weatherDescriptionCounts.sum())) #p(Ck)

#Calculating feature conditional probabilities 
humidityConditionalProbs = ((weatherDescriptionCounts.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'humidity'))
cloudcoverConditionalProbs = ((weatherDescriptionCounts.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'cloudcover'))
precipitationConditionalProbs = ((weatherDescriptionCounts.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'precip'))

print(numericalConditionalProb('Clear', 'temperature', 19))

#Read in test files
# testFileList = list()
# for file in pl.Path(sys.argv[2]).iterdir(): 
#         testFileList.append(file.name)

# numbers = re.compile(r'(\d+)')
# def numericalSort(value):
#     parts = numbers.split(value)
#     parts[1::2] = map(int, parts[1::2])
#     return parts

# #Make predictions
# testFileList.sort(key=numericalSort)

#COMMENT OUT BEFORE SUBMISSION
#Calculate accuracy in parallel
# truthFile = pd.read_json('ground_truth.json')
# accuracyIndex = 0
# accuracyCounter = 0
# for file in testFileList: 
#        testData = pd.read_excel('tests/' + file)
#        testHumidity = testData['humidity'][27]
#        testCloudCover = testData['cloudcover'][27]
#        testPrecipitation = testData['precip'][27]
#        prediction = classificationProb(testHumidity, testCloudCover, testPrecipitation).idxmax()
#        if prediction == (truthFile[0][accuracyIndex]):
#               accuracyCounter += 1
              
#        accuracyIndex +=1              
#        print(prediction)

# accuracy = accuracyCounter / 1000
# print(accuracy)

