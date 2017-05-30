'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from nltk.classify import scikitlearn
from statsmodels.sandbox.tsa.try_var_convolve import yvalid
Created on May 17, 2017

Project 2: Project 2 ADT analyzes a scikit-learn formatted dataset to create
           a classification report and confusion matrix. the confusion matrix
           is then turned into a matplotlib heatmap.

           (This ADT will eventually create a SKL dataset based on EGR-1 gene
           characteristics and social complexity criteria and measurements)

Dependencies: scikit-learn, matplotlib

Assumptions and Implementation Notes:
            -- All dependencies are installed and accessible


@authors: Jayse Farrel, Jessica Kunder, Ryan Palm
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

__author__ = "Jayse Farrel, Jessica Kunder, Ryan Palm"
__copyright__ = "COPYRIGHT_INFORMATION"
__credits__ = ["Camilo Acosta, Jayse Farrel, Jessica Kunder, Ryan Palm"]
__license__ = "GPL"
__version__ = "1.6.0dev"
__maintainer__ = "AUTHOR_NAME"
__email__ = "AUTHOR_EMAIL"
__status__ = "homework"

from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import pandas as pd



def main():
    """ Main Function to control program flow
    """
    tData = pd.read_csv('training_data_sheet.csv', index_col = 0)
    vData = pd.read_csv('validation_data_sheet.csv', index_col = 0)

#     modTrainData = modDataSet(tData)
#     modvData = modDataSet(vData)
    
    x_Train = getFeaturesList(tData)
    x_Valid = getFeaturesList(vData)
    
    y_Train = getTargetsList(tData)
    y_Valid = getTargetsList(vData)
    
    analyzeData(x_Train, y_Train, x_Valid, y_Valid, tData, vData)
    
    
    #learningCode()  #  Example code, basic regression and classification
    
    
    #initDataSets()


    return


def getFeaturesList(df):
    
    return list(df.columns[2:10])

def getTargetsList(df):
    
    return df['Species'].unique()


def analyzeData(x_Train, y_Train, x_Valid, y_Valid, dfTrain, dfValid):
    
    #x and y parameters for fitting and prediction
    xT = dfTrain[x_Train]
    yT = dfTrain["Gene_length"]
    
    xV = dfValid[x_Valid]
    yV = dfValid["Gene_length"]
    
    # Classifcation model
    model = DecisionTreeClassifier()
    
    # Train model with training data set
    model.fit(xT,yT)
    
    print(model)
    print()
    
    # expected (actual gene lengths)
    expected = yV
    # predict gene lengths based on features of valid set
    predicted = model.predict(xV)
    
    #print classification report
    cr = metrics.classification_report(expected, predicted)
    print(cr)
    
    #print confusion matrix
    cm = metrics.confusion_matrix(predicted, expected)
    print(cm)
    
    #Plot confusion matrix
    plotCM(cm, y_Valid) #  Plot the confusion matrix as a heat map

    return
    

def learningCode():
    """ This function is temporary as we learn how to use scikit-learn
        --This function includes a sample dataset provided by scikit-learn
        --There is an example of a regression and classification here.
    """

    """
    The iris data set contains 50 samples and two lists. A data array (of
    lists) and a target array.

    The target array is the "label" array. There were three iris species in
    the dataset, so we have 3 values in the target array [0,1,2]. Each value
    represents a species. The array index correlates with a specific data array
    index.

    the data array is an array of lists. Each array index correlates with it's
    respective label(target), and in this data set, there is a list with 4
    measurements: Sepal length, Sepal width, Petal length, and Petal width.
    """
    
    dataset = datasets.load_iris()

    model = DecisionTreeClassifier()        #  Create new DTC object
    """
    Fit a Classification and Regression Tree (CART) Model to the dataset.

    DecisionTreeClassifier().fit "trains" the model using the IRIS data set.
    this method provides functionality to add sample weights as well.
    """
    y = dataset.target
    x = dataset.data
    
    #model.fit(dataset.data, dataset.target)
    model.fit(x,y)
    
    #  http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    #  To assist in reading the DTC output
    print(model)
    print()

    #  Predictions
    expected = y # assign target (label) array to expected
    """
    http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.predict

    the DecisionTreeClassifier().predict function predicts class or regression
    value for each for each sample in parameter (parameter is an "array-like"
    or sparse matrix of shape = [n_samples, n_features].

    it returns a parameter-like array of predicted values
    """
    predicted = model.predict(x)

    #  Summarize the fit (relationship) of the model
    """
    How to interpret a Classification report:

    precision (positive predictive value): is the fraction of relevant
    instances among the retrieved instances.

    recall (sensitivity): is the fraction of relevant instances that
    have been retrieved over total relevant instances in the image

    f1-score: a measure of a test's accuracy. It considers both the
    precision and recall to compute the score. (The harmonic mean of
    precision and recall)

    """
    cr = metrics.classification_report(expected, predicted)
    print(cr)

    """
    How to interpret a Confusion Matrix: Y axis is actual Classes, X axis is
    classes predicted by the model. In this example, row 0 (top row); (0,0)
    is the number of correctly predicted classes. We have 50 samples in the
    Iris data set, and all 50 samples associated with label 0 were placed in
    label 0, hence (0,0) = 50. If we had 50 samples all which were actually in
    label 0, and 25 were correctly predicted to be in label 0, but 10 were
    predicted to be in label 1 and 15 were predicted to be in label 2 our row 0
    would look like this [25, 10, 15]

    In this data set, all predictions were 100% accurate.
    """
    cm = metrics.confusion_matrix(predicted, expected)
    print(cm)
    plotCM(cm) #  Plot the confusion matrix as a heat map

    return

def plotCM(cm, labels):
    """This method plots our confusion matrix as a heatmap with matplotlib
    """

    #labels = ['class 0', 'class 1', 'class 2'] #  Labels for heatmap

    """
    documentation on subplots:
    http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplots

    plt.subplots() is a function that returns a tuple (finite ordered list of
    elements) of a figure and axes object(s). fig is used for changing figure-
    level objects or save the figure as an image file. Ax is used to modify
    axes objects.
    """
    fig, ax = plt.subplots()

    # the matshow function returns a color matrix based on the data set
    h = ax.matshow(cm)

    # the colorbar function converts a color matrix into a image
    fig.colorbar(h)

    # Add labels to graph axes
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')

    # display the heat map
    plt.show()

    return

main()
