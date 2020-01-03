import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

#Function to show accuracy and losses plot
def print_cross_results(cross_result):
    score = cross_result['test_score']
    print('Scores =', score)
    print("Mean Accuracy: " + str(score.mean()))
    print("Standard deviation: " + str(score.std()))
    return


def print_classification_report(cross_result, data, labels):
    max_index = np.argmax(cross_result['test_score'])

    classifier = cross_result['estimator'][max_index]

    pred = classifier.predict(data);
    print("Classification Report")
    print(classification_report(labels, pred))
    print("Confusion Report")
    print(confusion_matrix(labels, pred))
    return

def check_cross_results_on_test(cross_result, data, labels):
    estimators = cross_result['estimator']
    for index in range(len(estimators)):
        print ("Index: " + str(index) + " \tResults: " + str(estimators[index].score(data, labels)))

def random_forest():
    train_file = pd.read_csv('mnist_train.csv')
    test_file = pd.read_csv('mnist_test.csv')

    x_train,y_train=train_file.iloc[:,1:].values,train_file.iloc[:,0].values
    x_test,y_test=test_file.iloc[:,1:].values,test_file.iloc[:,0].values

    #Conversion on array type
    X = np.asarray(x_train)
    y = np.asarray(y_train)

    X_t = np.asarray(x_test)
    y_t = np.asarray(y_test)


    #KLasyfikator lasu losowego - 10 drzew, drzewa maksymalnie głębokie
    clf = RandomForestClassifier(n_estimators=10, max_depth=None)
    cross_result = cross_validate(clf, X, y, cv=10, return_estimator=True)
    print_cross_results(cross_result)
    print_classification_report(cross_result, X_t, y_t)
    check_cross_results_on_test(cross_result, X_t, y_t)

    rfc = RandomForestClassifier()

    # max_feaures 'auto' = 'sqrt'
    param_grid = {
        'n_estimators': [10, 20, 30, 40, 50, 100, 200, 300, 400],
        'max_features': ['auto', 'log2']
    }
    classifier = RandomForestClassifier()

    #poszukiwanie parametrów
    CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    CV_rfc.fit(X, y)
    print(CV_rfc.best_params_)

#    RF = RandomForestClassifier(max_features='auto', n_estimators=200)
#    cross_result = cross_validate(RF, X, y, cv=10, return_estimator=True)
#    print_cross_results(cross_result)
#    print_classification_report(cross_result, X_t, y_t)
#   check_cross_results_on_test(cross_result, X_t, y_t)