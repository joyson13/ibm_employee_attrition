from sklearn.metrics import accuracy_score, confusion_matrix

def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    
    # checking if model has method called predict_proba
    if  hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
        # predict_proba return a matrix of two column (negative and positives), [:,1] means probability
    else:
        # if model doesn't have predict_prba it uses decision_tree to test eacj instance
        y_prob = model.decision_function(X_test)
        
    return accuracy, cm, y_test, y_prob