from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def predict(model, X_test, y_test):
    predictions = model.predict(X_test)

    conf_matrix = confusion_matrix(y_test, predictions)

    print("Confusion matrix")
    print(conf_matrix)
    print("Classification report")
    print(classification_report(y_test, predictions))
    print("Accuracy: " + str(accuracy_score(y_test, predictions)))
