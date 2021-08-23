#!/bin/env python3

### Set up the env ###
# Import your basic libraries
import pandas as pd
import numpy as np
from termcolor import colored
from tqdm import tqdm
import math, time

# Import libraries for KNN modeling
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# Import libraries for Random Forest modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

# Import libraries for ANN
from sklearn.neural_network import MLPClassifier



### Build a function to containerize the cleaning of data ###
def cleanData(roundDF, colNames):
    # Convert the strings in 'map' attribute to numerical values
    tmpMaps = []
    for i in roundDF['map']:
        if i not in tmpMaps:
            tmpMaps.append(i)
    
    mapVals = list(range(1,len(tmpMaps)+1)) # Formerly a list of values 1-8

    tmpBomb = [False, True]
    bombVals = [0, 1]

    roundDF.replace(to_replace = tmpMaps, value = mapVals, inplace=True)
    roundDF.replace(to_replace = tmpBomb, value = bombVals, inplace=True)    

    # Split the rounds into two different dataframes: attributes and target
    x = roundDF[colNames[3:-1]] # Attributes
    y = roundDF['round_winner'] # Target

    return x, y


### Build a function to group the snapshots into rounds ###
def buildRounds(csgoData):
    # Set up a new list to hold all of the round summaries
    rounds = []

    # Set up loop params to summarize rounds
    counter = 0
    lim = len(csgoData) # This will be set to len(csgoData) once it is ready
    startPoint = 0
    
    # Iterate through the data set to slice the rounds
    print(colored('Grouping snapshots into rounds...', 'yellow'))
    pbar = tqdm(total=lim)
    while counter < lim + 1:

        roundOver = 'FALSE'
        prev = 180

        # Loop for only the snapshots within the round
        while roundOver == 'FALSE':
            if counter == lim:
                roundSum = []
                roundSlice = csgoData.loc[startPoint:counter-1]

                # print('Round Over - %s won the round' % csgoData['round_winner'][counter]) # -> Used to determine if the round has been completed (new round started) # -> Debug line
                
                # Some debugging lines
                '''
                print(roundSlice) -> This is to validate that the rounds are being sliced properly
                print(roundSlice.iloc[0]) -> Used to test if I can pull the first row of the slice
                print(roundSlice.iloc[len(roundSlice)-1]) -> Used to test if I can pull the last row of the slice
                '''

                # Set some of the values according to the end of the round
                roundSum.append(roundSlice.iloc[len(roundSlice)-1]['ct_score'])
                roundSum.append(roundSlice.iloc[len(roundSlice)-1]['t_score'])
                roundSum.append(roundSlice.iloc[len(roundSlice)-1]['map'])
                roundSum.append(roundSlice.iloc[len(roundSlice)-1]['bomb_planted'])
                
                # Set the remaining values according to the beginning of the round
                for i in roundSlice.iloc[1][5:97]:
                    roundSum.append(i)
                
                # Test to see if there is a short round that indicates inaccurate data capture
                if len(roundSlice) <= 2 and roundSum[0] == rounds[len(rounds)-1][0] and roundSum[1] == rounds[len(rounds)-1][1]:
                    print('Round %s Triggered: Short round - Ignoring it...\n' % counter)
                else:
                    rounds.append(roundSum)

                del roundSum

                pbar.close()
                return rounds

            elif counter < lim:
                if csgoData['time_left'][counter] > prev and csgoData['time_left'][counter] > 165:
                    
                    roundSum = []
                    roundSlice = csgoData.loc[startPoint:counter-1]

                    # print('Round Over - %s won the round' % csgoData['round_winner'][counter]) # -> Used to determine if the round has been completed (new round started) # -> Debug line
                    
                    # Some debugging lines
                    '''
                    print(roundSlice) -> This is to validate that the rounds are being sliced properly
                    print(roundSlice.iloc[0]) -> Used to test if I can pull the first row of the slice
                    print(roundSlice.iloc[len(roundSlice)-1]) -> Used to test if I can pull the last row of the slice
                    '''

                    # Set some of the values according to the end of the round
                    roundSum.append(roundSlice.iloc[len(roundSlice)-1]['ct_score'])
                    roundSum.append(roundSlice.iloc[len(roundSlice)-1]['t_score'])
                    roundSum.append(roundSlice.iloc[len(roundSlice)-1]['map'])
                    roundSum.append(roundSlice.iloc[len(roundSlice)-1]['bomb_planted'])
                    
                    # Set the remaining values according to the beginning of the round
                    for i in roundSlice.iloc[1][5:97]:
                        roundSum.append(i)
                    
                    # Test to see if there is a short round that indicates inaccurate data capture
                    if len(roundSlice) <= 2 and roundSum[0] == rounds[len(rounds)-1][0] and roundSum[1] == rounds[len(rounds)-1][1]:
                        print('Round %s Triggered: Short round - Ignoring it...\n' % counter)
                    else:
                        rounds.append(roundSum)

                    del roundSum

                    roundOver = 'TRUE'
                    startPoint = counter  
                
                # Test for repeated rows in the data
                elif csgoData['time_left'][counter] == prev:
                    # print('Repeat row - ignoring...') # -> Debug line
                    prev = csgoData['time_left'][counter]

                # Round not over yet, so set the new reference time value for the next iteration
                else:
                    # print('Round not over. Time remaining: %s' % csgoData['time_left'][counter]) # -> Used to test if the round timer was being referenced properly
                    prev = csgoData['time_left'][counter]
                
                counter += 1
                pbar.update(1)

### Create a function for applying the KNN model ###
def useKNN(roundDF, colNames):
    # Convert the data to necessary values and split it into target and attributes
    x, y = cleanData(roundDF, colNames)

    # Split the data into training and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)
    scaler = StandardScaler()
    scaler.fit(xTrain)

    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)

    # Apply the KNN methodology using the training sets for attributes and taget starting with k as the square root of the length of the training set
    print(colored('\nPreliminary KNN Model with k=10', 'yellow'))
    k = int(math.sqrt(len(xTrain)))
    classifier = KNeighborsClassifier(n_neighbors=10)
    classifier.fit(xTrain, yTrain)

    yPrediction = classifier.predict(xTest)

    # Create a confusion matrix for the results
    confMatrix = confusion_matrix(yTest, yPrediction)
    
    # Format and print the confusion matrix
    confMatrixCols = ['CT', 'T']
    knnConfMatrix = pd.DataFrame(confMatrix, columns =confMatrixCols)
    
    print('\n',knnConfMatrix)
    
    # Print the report from the classification results
    print(classification_report(yTest, yPrediction))

    # Find the best k value based on error rates
    error = []

    print(colored('Finding best k value based on error rates...', 'yellow'))
    lim = int(math.sqrt(len(xTrain)))
    #pbar = tqdm(total=lim)
    
    kRange = list(range(1,lim))

    for i in kRange:
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xTrain, yTrain)
        iPrediction = knn.predict(xTest)
        error.append(np.mean(iPrediction != yTest))

        print('K=%s' % i)
        knnTrainScore = knn.score(xTrain, yTrain) * 100
        print('KNN Model Train Accuracy: %s' % knnTrainScore)

        knnTestScore = knn.score(xTest, yTest) * 100
        print('KNN Model Test Accuracy: %s' % knnTestScore)
        
        
        #pbar.update(1)
    #pbar.update(1)
    #pbar.close()

    # Calculate the error values to find the best k value
    lowest = [1, 1]
    for i in kRange:
        pair = (i, error[i-1])
        if error[i-1] < lowest[1]:
            lowest[0] = i
            lowest[1] = error[i-1]
    
    # Print the best k value with the error rate
    bestK = (lowest[0], lowest[1])
    print(colored('\nBest K: %s\tError Rate: %s' % bestK))

    # Apply the KNN methodology using the training sets for attributes and taget with k set to the calculated best k
    print(colored('\nKNN Model with Best K', 'yellow'))
    k = int(math.sqrt(len(xTrain)))
    classifier = KNeighborsClassifier(n_neighbors=bestK[0])
    classifier.fit(xTrain, yTrain)

    knnTrainScore = classifier.score(xTrain, yTrain) * 100
    print('KNN Model Train Accuracy: %s' % knnTrainScore)

    knnTestScore = classifier.score(xTest, yTest) * 100
    print('KNN Model Test Accuracy: %s' % knnTestScore)

    yPrediction = classifier.predict(xTest)

    # Create a confusion matrix for the results
    confMatrix = confusion_matrix(yTest, yPrediction)
    
    # Format and print the confusion matrix
    confMatrixCols = ['CT', 'T']
    knnConfMatrix = pd.DataFrame(confMatrix, columns =confMatrixCols)
    
    print('\n',knnConfMatrix)
    
    # Print the report from the classification results
    print(classification_report(yTest, yPrediction))

    # Print the summary for the results
    summaryK = bestK[0]
    summaryError = bestK[1]
    summaryError = '{0:.2f}'.format(summaryError)
    print('\n\nUsing K Nearest Neighbor with k =', colored(summaryK, 'yellow'), ', the model can predict which team will win a round with a', colored(summaryError, 'yellow'),'% error rate.\n')

    return

### Create a function to use Random Forests to make a prediction model ###
def randomForest(roundDF, colNames):
    # Convert the data to necessary values (numeric) and split it into target and attributes
    toNum = {'CT': 0, 'T': 1}
    roundDF.replace({'CT': toNum, 'T': toNum})
    
    # Clean the data to split it
    x, y = cleanData(roundDF, colNames)
          
    # Split the data into training and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)

    # Build the forest using pre-determined tree counts
    print(colored('Calculating Random Forests...', 'yellow'))
    rfResults = []
    nTrees = [1, 5, 10, 20, 50, 100, 500, 1000, 3000, 5000]
    for i in tqdm(nTrees):
        time.sleep(2)
        rf = RandomForestClassifier(n_estimators=i)
        rf.fit(xTrain, yTrain)

        # Create a prediction using the model
        yPrediction = rf.predict(xTest)

        # Print accuracy rate of preliminary forest
        rfAccuracy = metrics.accuracy_score(yTest, yPrediction) * 100
        rfAccuracy = '{0:.3f}'.format(rfAccuracy)
        tmpResult = [i, rfAccuracy]
        rfResults.append(tmpResult)

        time.sleep(2)
    
    # Store the Random Forest results into a new data frame to compare the accuracy of each tree count    
    rfResults = pd.DataFrame(rfResults, columns = ['Trees', 'Accuracy'])
    print(rfResults)

    # Find the best tree count based on accuracy
    lim = list(range(0, len(rfResults)))
    bestTrees = [1, 0.0]

    for i in lim:
        tmpAcc = rfResults.iloc[i]['Accuracy']
        tmpAcc = float(tmpAcc)
        if tmpAcc >= float(bestTrees[1]):
            bestTrees[0] = rfResults.iloc[i]['Trees']
            bestTrees[1] = rfResults.iloc[i]['Accuracy']
        
        pPair = (colored(bestTrees[0], 'yellow'), colored(bestTrees[1] + '%', 'yellow'))
    
    # Display the best tree count to use based on the results
    print(colored('\nResults of Random Forests model', 'yellow'))
    print('Best Trees: %s\nAccuracy: %s' % pPair)

    rfTrainScore = rf.score(xTrain, yTrain) * 100
    print('RF Train Accuracy: %s' % rfTrainScore)

    rfTestScore = rf.score(xTest, yTest) * 100
    print('RF Test Accuracy: %s' % rfTestScore)

    return



### Function for Artificial Neural Net ###
def ann(roundDF, colNames):
    # Clean the data to split it
    x, y = cleanData(roundDF, colNames)
          
    # Split the data into training and test sets
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 42)
    
    # Apply the data to the ANN model
    annModel = MLPClassifier(hidden_layer_sizes=(80, 40, 20))
    annModel.fit(xTrain,yTrain)

    # Score the ANN model
    annAccuracy = annModel.score(xTest, yTest)*100
    annAccuracy = '{0:.2f}'.format(annAccuracy)

    # Print the accuracy of the ANN model
    print('Accuracy of ANN model: %s' % colored(annAccuracy, 'yellow'))

    return
    

### Main function to route data to other functions ###
def main():
    # Import the csv file and view the head of the data
    csgoData = pd.read_csv('csgo_round_snapshots.csv')
    
    # Check for NA values
    print(colored('Checking for NA values...', 'yellow'))
    
    isNA = csgoData.isnull().sum()
    naCounter = 0

    # Check each isNA value individually
    pbar = tqdm(total=len(isNA))
    for i in isNA:
        if i > 0:
            print('NA: ',i)
            naCounter += i
        pbar.update(1)
    pbar.close()
    
    # Print NA value count
    print('%s NA values found.\n\n' % colored(naCounter, 'yellow'))
    
    # Grab a list of the column names
    colNames = []
    for col in csgoData.columns:
        colNames.append(col)
    
    # Slice the data into rounds with the 'buildRounds' function
    rounds = buildRounds(csgoData)

    # Print some round summaries
    print(colored('\n----- Data Frame Preview -----\n', 'yellow'))
    print('\nNumber of rounds: %s' % colored(len(rounds), 'yellow'))
    
    # Show the new dataframe consisting of round summaries
    roundDF = pd.DataFrame(rounds, columns = colNames[1:97])
    print(roundDF)

    # Send data to be applied to KNN model
    print(colored('\n----- K Nearest Neighbor -----\n', 'yellow'))
    useKNN(roundDF, colNames)

    # Send data to be applied to Random Forests model
    print(colored('\n----- Random Forests -----\n', 'yellow'))
    randomForest(roundDF, colNames)

    # Send data to be applied to ANN model
    print(colored('\n----- Artificial Neural Net -----\n', 'yellow'))
    ann(roundDF, colNames)

### Start the script ###
main()