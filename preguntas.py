import pandas as pd

def pregunta_01():
    df = pd.read_csv("mushrooms.csv")
    df = df.drop("veil_type", axis=1)
    y = df.type
    X = df.copy()
    X = X.drop('type', axis=1)
    return X, y

def pregunta_02():
    from sklearn.model_selection import train_test_split
    X, y = pregunta_01()
    (X_train, X_test, y_train, y_test,) = train_test_split(X, y, test_size=50, random_state=123)
    return X_train, X_test, y_train, y_test

def pregunta_03():
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.pipeline import Pipeline
    X_train, _, y_train, _ = pregunta_02()
    pipeline = Pipeline(steps=[("OneHotEncoder", OneHotEncoder()), ("LogisticRegressionCV", LogisticRegressionCV(Cs=10))])
    pipeline.fit(X_train, y_train)
    return pipeline

def pregunta_04():
    from sklearn.metrics import confusion_matrix
    pipeline = pregunta_03()
    X_train, X_test, y_train, y_test = pregunta_02()
    cfm_train = confusion_matrix(y_true=y_train, y_pred=pipeline.predict(X_train))
    cfm_test = confusion_matrix(y_true=y_test, y_pred=pipeline.predict(X_test))
    return cfm_train, cfm_test

