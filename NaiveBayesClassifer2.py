import sys
import pandas as pd
import numpy as np
import pathlib as pl
import os
import re
import math

trainingData = pd.read_excel(sys.argv[1])


#Calculate probabilities of each class
classificationProbs = (trainingData['weather_descriptions']).value_counts(normalize=True)
# Clear                             0.349617
# Sunny                             0.345238
# Partly cloudy                     0.134784
# Cloudy                            0.059113
# Overcast                          0.048987
# Patchy rain possible              0.025315
# Moderate or heavy rain shower     0.018473
# Moderate rain at times            0.009168
# Heavy rain at times               0.006705
# Light freezing rain               0.001779
# Patchy moderate snow              0.000684
# Moderate or heavy snow showers    0.000137

#Calculate conditional probabilities
#Categorical 
def categoricalConditionalProbs(feature, classification): #P(y|x)
    givenData = trainingData.loc[trainingData['weather_descriptions'].shift(-1) == classification]
    return givenData[feature].value_counts(normalize=True)

def numericalConditionalProbs(feature, classification): 
    givenData = trainingData.loc[trainingData['weather_descriptions'].shift(-1) == classification]
    mu_k = givenData[feature].mean
    sigma_k = givenData[feature].std
    return 

#Calculate necessary categorical conditional probabiliites
descriptionProbs = (classificationProbs.index.to_series()).apply(lambda x : categoricalConditionalProbs('weather_descriptions', x))
descriptionProbs = descriptionProbs.fillna(0)
precipitationProbs = (classificationProbs.index.to_series()).apply(lambda x : categoricalConditionalProbs('precip', x))
precipitationProbs = precipitationProbs.fillna(0)
humidityProbs = (classificationProbs.index.to_series()).apply(lambda x : categoricalConditionalProbs('humidity', x))
humidityProbs = humidityProbs.fillna(0)

#Prediction function
def predict(prevDescription, precipitation, humidity):
      if(prevDescription != 'Moderate snow'):
        possiblePredictions = (classificationProbs.index.to_series()).apply(lambda x: classificationProbs[x] * descriptionProbs[prevDescription][x] * precipitationProbs[precipitation][x] * humidityProbs[humidity][x])
        return possiblePredictions.idxmax()
      else:
        return 'Clear'

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
truthFile = pd.read_json('ground_truth.json')
accuracyIndex = 0
accuracyCounter = 0
for file in testFileList: 
       testData = pd.read_excel('tests/' + file)
       testDescriptions = testData['weather_descriptions'][27]
       testPrecipitation = testData['precip'][27]
       testHumidity = testData['humidity'][27]
       prediction = predict(testDescriptions, testPrecipitation, testHumidity)
       if prediction == (truthFile[0][accuracyIndex]):
              accuracyCounter += 1
              
       accuracyIndex +=1              
       #print(prediction)
accuracy = accuracyCounter / 1000
print(accuracy)


