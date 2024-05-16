import sys
import pandas as pd
import numpy as np
import pathlib as pl
import os
import re
import math

trainingData = pd.read_excel(sys.argv[1])

def categoricalConditionalProb(category, descriptor, shift):
        tempProbs = pd.Series(trainingData.loc[trainingData['weather_descriptions'].shift(periods=-(shift)) == category, descriptor].value_counts(normalize=True)) #shift so that we are checking for FUTURE weather condition
        return tempProbs

def numericalConditionalProb(category, descriptor, value, shift): 
        categoryData = pd.Series(trainingData.loc[trainingData['weather_descriptions'].shift(periods=-(shift)) == category, descriptor]) #Pulls temperatures for which next day is given category
        return (1 / (math.sqrt(2*math.pi*(math.pow(categoryData.std(), 2))))*(math.pow(math.e, -((math.pow(value - categoryData.mean(), 2))/(2*(math.pow(categoryData.std(), 2)))))))        

def conditionalProb(classification, description, humidity, cloudcover, precipitation, temperature, pressure, description2, description3, description4, description5): #pull conditional probabilities for given attributes and return probability for each classification
        classProb = weatherDescriptionProbs[classification]
        descriptionProb = 1
        descriptionProb2 = 1
        descriptionProb3 = 1
        descriptionProb4 = 1
        descriptionProb5 = 1
        if(description in descriptionConditionalProbs.index):
                descriptionProb += descriptionConditionalProbs[description][classification]
        if(description2 in descriptionConditionalProbs2.index):
                descriptionProb2 += descriptionConditionalProbs2[description2][classification]
        if(description3 in descriptionConditionalProbs3.index):
                descriptionProb3 += descriptionConditionalProbs3[description3][classification]
        if(description4 in descriptionConditionalProbs4.index):
                descriptionProb4 += descriptionConditionalProbs4[description4][classification]
        if(description5 in descriptionConditionalProbs5.index):
                descriptionProb5 += descriptionConditionalProbs5[description5][classification]
        humidityProb = humidityConditionalProbs[humidity][classification]
        cloudcoverProb = cloudcoverConditionalProbs[cloudcover][classification]
        precipitationProb = precipitationConditionalProbs[precipitation][classification]
        temperatureProb = numericalConditionalProb(classification, 'temperature', temperature, 1)
        pressureProb = numericalConditionalProb(classification, 'pressure', pressure, 1)

        return classProb*(descriptionProb * humidityProb * cloudcoverProb * precipitationProb * temperatureProb * pressureProb * descriptionProb2 * descriptionProb3 * descriptionProb4 * descriptionProb5)

def classificationProb(description, humidity, cloudcover, precipitation, temperature, pressure, description2, description3, description4, description5):
        #Previous day 
        oneDayBefore = (weatherDescriptionProbs.index).to_series().apply(lambda x : conditionalProb(x, description, humidity, cloudcover, precipitation, temperature, pressure, description2, description3, description4, description5))
        return oneDayBefore
                                                                   
#Calculate classification probabilities
weatherDescriptionProbs = pd.Series(trainingData['weather_descriptions'].value_counts(normalize=True))

#Calculating categorical feature conditional probabilities 
descriptionConditionalProbs = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'weather_descriptions', 1))
humidityConditionalProbs = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'humidity', 1))
cloudcoverConditionalProbs = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'cloudcover', 1))
precipitationConditionalProbs = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'precip', 1))

descriptionConditionalProbs2 = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'weather_descriptions', 2))
descriptionConditionalProbs3 = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'weather_descriptions', 3))
descriptionConditionalProbs4 = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'weather_descriptions', 4))
descriptionConditionalProbs5 = ((weatherDescriptionProbs.index).to_series()).apply(lambda x : categoricalConditionalProb(x, 'weather_descriptions', 5))

#Read in test files
testFileList = list()
for file in pl.Path(sys.argv[2]).iterdir(): 
        testFileList.append(file.name)

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# #Make predictions
testFileList.sort(key=numericalSort)

#COMMENT OUT BEFORE SUBMISSION
#Calculate accuracy in parallel
# truthFile = pd.read_json('ground_truth.json')
# accuracyIndex = 0
# accuracyCounter = 0
for file in testFileList: 
       testData = pd.read_excel(sys.argv[2] +'/' + file)
       testDescriptions = testData['weather_descriptions'][27]
       testHumidity = testData['humidity'][27]
       testCloudCover = testData['cloudcover'][27]
       testPrecipitation = testData['precip'][27]
       testTemperature = testData['temperature'][27]
       testPressure = testData['pressure'][27]

       testDescriptions2 = testData['weather_descriptions'][26]
       testDescriptions3 = testData['weather_descriptions'][25]
       testDescriptions4 = testData['weather_descriptions'][24]
       testDescriptions5 = testData['weather_descriptions'][23]


       prediction = classificationProb(testDescriptions, testHumidity, testCloudCover, testPrecipitation, testTemperature, testPressure, testDescriptions2, testDescriptions3, testDescriptions4, testDescriptions5).idxmax()
#        if prediction == (truthFile[0][accuracyIndex]):
#               accuracyCounter += 1
              
#        accuracyIndex +=1              
       print(prediction)
# accuracy = accuracyCounter / 1000
# print(accuracy)



