## Used to calculate metric for the models.
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn import metrics
import pandas as pd

def print_metrics(clf, X_test, y_test):

    y_pred = clf.predict(X_test)
    print("Confusion matrix\n")
    print(pd.crosstab(pd.Series(y_test, name='Actual'), pd.Series(y_pred, name='Predicted')))
    accuracy, precision, recall, f1 = get_metrics(y_test, y_pred)
    print("accuracy = %.3f \nprecision = %.3f \nrecall = %.3f \nf1 = %.3f" % (accuracy, precision, recall, f1))
    
    return accuracy, precision, recall, f1
    
def get_metrics(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1