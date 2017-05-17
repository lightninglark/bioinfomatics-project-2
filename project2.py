'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
Created on May 17, 2017

Project 2: (Project 2 ADT definition)

Dependencies: scikit-learn

Assumptions and Implementation Notes:
            -- 


@authors: Camilo Acosta, Jayse Farrel, Jessica Kunder, Ryan Palm
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

__author__ = "Camilo Acosta, Jayse Farrel, Jessica Kunder, Ryan Palm"
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

def main():
    """ Main Function to control program flow
    """
    
    learningCode()  # Example code, basic regression and classification
    
    return

   
def learningCode():
    """ This function is temporary as we learn how to use scikit-learn
        --This function includes a sample dataset provided by scikit-learn
        --There is an example of a regression and classification here.
    """ 
    dataset = datasets.load_iris()          # Load sample iris (flower) dataset
    
    model = DecisionTreeClassifier()        # Create new DTC object
    
    #Fit a Classification and Regression Tree (CART) Model to the dataset
    model.fit(dataset.data, dataset.target)
    print(model)
    
    #Predictions
    expected = dataset.target
    predicted = model.predict(dataset.data)
    
    #Summarize the fit (relationship) of the model
    print(metrics.classification_report(expected, predicted))     
    print(metrics.confusion_matrix(predicted, expected))
    
    return
      
    
main()

