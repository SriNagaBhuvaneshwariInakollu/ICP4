import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier


train_df = pd.read_csv('./train_preprocessed.csv')
test_df = pd.read_csv('./test_preprocessed.csv')
X_train = train_df["Sex"]
Y_train = train_df["Survived"]
relation = train_df[["Survived","Sex"]].groupby(['Sex'],as_index = False).mean()
print (relation)