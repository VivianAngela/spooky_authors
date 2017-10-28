import pandas as pd
def predict_for_kaggle(clf, X, columns, ids, name):
    """Predict class probabilites for Kaggle submission
    
    Args:
        clf      (Pipeline): sklearn Pipeline
        X          (Series): test phrases to be predicted
        columns      (list): class labels (e.g. EAP, HPL, MWS)
        ids        (Series): IDs of test phrases
        name       (String): name of the submission
    
    Return:
        predictions    (df): dataframe with class probabilities for each test phrases
    
    """
    
    predictions = pd.DataFrame(clf.predict_proba(X), columns=columns, index=ids)
    
    predictions.to_csv('submissions/' + name + ".csv")
    
    return predictions