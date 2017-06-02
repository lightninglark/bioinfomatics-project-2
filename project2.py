'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from nltk.classify import scikitlearn
from statsmodels.sandbox.tsa.try_var_convolve import yvalid
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
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
            -Credit to Kevin Palm for cluster analysis code
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

__author__ = "Jayse Farrel, Jessica Kunder, Ryan Palm"
__copyright__ = "COPYRIGHT_INFORMATION"
__credits__ = ["Camilo Acosta, Jayse Farrel, Jessica Kunder, Ryan Palm"]
__license__ = "GPL"
__version__ = "1.6.0dev"
__maintainer__ = "AUTHOR_NAME"
__email__ = "AUTHOR_EMAIL"
__status__ = "homework"

from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn import cluster
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np



def main():
    """ Main Function to control program flow
    """
    tData = pd.read_csv('./input/training_data.csv', index_col = 0)  # Training data
    vData = pd.read_csv('./input/validation_data.csv', index_col = 0)# Validation data
    fData = pd.read_csv('./input/full_data.csv', index_col = 0)      # Full data

    # Grab features list and gene lengths for use in analysis/plots
    x_Full = getFeaturesList(fData)
    y_Full = getTargetsList(fData)

    x_Train = getFeaturesList(tData)
    x_Valid = getFeaturesList(vData)

    y_Train = getTargetsList(tData)
    y_Valid = getTargetsList(vData)

    # Plot our dataset (feature, genelength) (x,y)
    visualizeData(fData)

    #Run our DTR analysis, generate scatter plot (predicted, expected) (x,y)
    decisionTreeRegression(x_Train, y_Train, x_Valid, y_Valid, tData, vData)

    #Have not been able to plot yet
    clusterAnalysis(fData)

    return

def visualizeData(df):
    '''
    This function prints out scatter plots of all 'features' by gene length and saves them to the drive.
    '''
    x_Features = getFeaturesList(df)
    y_GeneLength =  getTargetsList(df)

    gl = 'Gene_length'

    for n in range(0, len(x_Features)):

        plt.figure(figsize = (4,3))

        plt.scatter(df[x_Features[n]], df[gl], marker = 'o', color = 'green', alpha = 0.7, s = 50, label = x_Features[n])

        plt.legend()

        plt.savefig('images\\' + x_Features[n] + '.png')


    return

def getFeaturesList(df):
    '''
    This function returns a list of the 'features' (like Monogmous, Solitary, etc.)
    '''
    return list(df.columns[1:10])

def getTargetsList(df):
    '''
    This function returns a list of all gene lengths.
    '''
    return df['Gene_length'].unique()

def clusterAnalysis(df):
    '''
    @author: Kevin Palm
    This function preforms a principle components clustering analysis.
    '''
    from sklearn.decomposition import PCA
    import matplotlib.cm as cm

    # Separate out features and outcomes
    x_Features = df[getFeaturesList(df)]
    y = df['Gene_length'].astype(float)

    # Lets use Principle Component Analysis to reduce our feature dimensions down to two for nice visualizations and broader/inclusive clusters
    pca = PCA(n_components=2)
    x_components = pca.fit_transform(x_Features)

    print("The PCA components preserve about {}% of the explained variance from the original dataset".format(int(sum(pca.explained_variance_ratio_)*100)))

    # Using kmeans as our clustering algorithm, but we need to pick the best number of clusters
    best_score = -1.0
    best_model = None
    best_tuning = None
    for i in range(2, len(df.index)):

        # Fit a kmeans model at this n_cluster size and get a silhouette score
        kmeans = cluster.KMeans(n_clusters=i)
        labels = kmeans.fit_predict(x_components)
        score = metrics.silhouette_score(x_components, labels)

        # Save if improvement
        if score > best_score:
            best_score = score
            best_model = kmeans
            best_tuning = i

    print("Using {0} clusters in our final model - which scored the highest silhouette score of {1}".format(best_tuning, round(best_score, 3)))

    # Format a dataframe for plotting clusters
    cluster_df = pd.DataFrame(x_components, columns=["Principle Component 1", "Principle Component 2"])
    cluster_df["Cluster Label"] = "Cluster " + pd.Series(best_model.predict(x_components)).astype(str)
    cluster_df["Length Scaler"] = (y-y.min())/(y.max()-y.min())*100

    # Also prepare series for examining principle component coeficients (to help with data interpretation)
    pc1_series = pd.Series(pca.components_[0], index=getFeaturesList(df)).sort_values()
    pc2_series = pd.Series(pca.components_[1], index=getFeaturesList(df)).sort_values()

    # Plot clusters along principle component space, with subplots of component coeficients to help with interpretation
    fig1, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={'width_ratios': [4, 2, 2]}) # define three subplots, one big two small
    clusterlist = list(cluster_df["Cluster Label"].unique()) # list of cluster labels
    for c in clusterlist:
        color = cm.gist_ncar(clusterlist.index(c)/len(clusterlist)) # pull a new color from the color map each iteration
        size = cluster_df[cluster_df["Cluster Label"]==c]["Length Scaler"] # we'll size each dot proportionally to the gene length
        cluster_df[cluster_df["Cluster Label"]==c].plot.scatter(x="Principle Component 1", y="Principle Component 2", color=color, label=c, ax=ax0, edgecolors='none', s=size, title="Feature Principle Components, Colored by\nCluster Label and Sized by Gene Length") # plot the cluster
    ax0.legend(loc=2, prop={'size': 6}) # move legend out of the way

    # Now let's plot the principle components to help with interpreting the feature space of our graph
    pc1_series.plot.barh(ax=ax1, title="Principle Component 1\nCoefficients")
    pc2_series.plot.barh(ax=ax2, title="Principle Component 2\nCoefficients")
    for ax in [ax1, ax2]: # Resizing output to all fit in
        for item in ([] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(8)
        ax.set_yticklabels(ax.yaxis.get_majorticklabels(), rotation=65)

    # Save to file
    fig1.savefig('images\\pca_clusters.png')

    # Let's also print the cluster centers back in the original feature space
    centers = best_model.cluster_centers_ # centers in pca space
    centers = pca.inverse_transform(centers) # centers in original feature space
    centers = pd.DataFrame(centers, columns=x_Features.columns) # Realign column titles
    centers.index = "Cluster " + centers.index.astype(str) # Better index
    print(centers)


    return

def decisionTreeRegression(x_Train, y_Train, x_Valid, y_Valid, dfTrain, dfValid):
    '''
    This function performs a supervised learning classification analysis using
    a decision tree regressor. It also prints applicable metrics
    '''

    #x and y parameters for fitting and prediction
    xT = dfTrain[x_Train]
    yT = dfTrain["Gene_length"]

    xV = dfValid[x_Valid]
    yV = dfValid["Gene_length"]



    # Regressor model
    model = DecisionTreeRegressor()

    # Train model with training data set
    model.fit(xT,yT)

    print(model)
    print()

    # expected (actual gene lengths)
    expected = yV
    # predict gene lengths based on features of valid set
    predicted = model.predict(xV)

    #Explained Variance Score

    expVariScore = metrics.explained_variance_score(expected, predicted)

    print('Explained Variance Score')
    print(expVariScore)
    print()

    #mean absolute error

    mAbsError = metrics.mean_absolute_error(expected, predicted)
    print('Mean Absolute Error')
    print(mAbsError)
    print()

    #mean squared error
    mSqrError = metrics.mean_squared_error(expected, predicted)
    print('Mean Squared Error')
    print(mSqrError)
    print()

    #median_absolute_erro
    medAbsError = metrics.median_absolute_error(expected, predicted)
    print('Median Absolute Error')
    print(medAbsError)
    print()

    #r2 score
    r2Score = metrics.r2_score(expected, predicted)
    print('r2 Score')
    print(r2Score)
    print()



    # Genrate stats summary (linear regression)
    statistics(expected, predicted)

    # Generate scatter plot
    plotDTR(expected, predicted)

    return

def statistics(expected,predicted):
    '''
    This method runs a linear regression (OLS: least squared) on the
    predicted (Y) vs true (X) gene lengths.
    '''

    results = sm.OLS(predicted, sm.add_constant(expected)).fit()

    print(results.summary())

    return

def plotDTR(expected, predicted):
    '''
    This function plots the results from our DTR analysis as a scatter plot.
    '''
    plt.figure()
    plt.scatter(expected,predicted, c= "darkorange", label = "Gene Length"  )

    #Add line of best fit
    plt.plot(np.unique(expected), np.poly1d(np.polyfit(expected, predicted, 1))(np.unique(expected)))


    plt.xlabel("Actual Gene Length (Nucleotide Base Pairs)")
    plt.ylabel("Predicted Gene Length (Nucleotide Base Pairs)")
    plt.title("Decision Tree Regression")
    plt.legend()

    plt.savefig('images/DTR_scatter.png')

    return
'''
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
'''

main()
