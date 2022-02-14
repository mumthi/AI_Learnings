from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def distribute_input(n_samples=1000, n_features=100, n_informative=10, n_redundant=90, random_state=1):
    X, y = make_classification(n_samples, n_features, n_informative=n_informative, n_redundant=n_redundant,
                               random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
    t = MinMaxScaler()
    t.fit(X_train)
    X_train = t.transform(X_train)
    X_test = t.transform(X_test)
    return X_train, X_test
