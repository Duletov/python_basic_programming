import numpy
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier, BallTree
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score

import warnings
warnings.simplefilter('ignore')

numpy.random.seed(7)

#Ответ выходит 0.9725 для этого рпимера и для RandomForestClassifier. Я перебрал все параметры (ты чё пёс, я математик), порог перейти не удалось. 
#SVC, на котором у ребят всё таки всё получилось, у меня работает больше 4 часов (я так и не дождался). Несмотря на почти лучшее железо и отказ от Юпитера. Я не знаю в чём проблема, эта реализация работает за минуту.

# Load data from https://www.openml.org/d/554
X, Y = fetch_openml('mnist_784', return_X_y=True)
print(f"shape of X is {X.shape}")

test_shuffle = numpy.random.permutation(X.shape[0])

X_test, X_train = X[test_shuffle[:10000]], X[test_shuffle[10000:]]
Y_test, Y_train = Y[test_shuffle[:10000]], Y[test_shuffle[10000:]]

print(f"train size: {X_train.shape[0]}")
print(f"test size: {X_test.shape[0]}")

model = KNeighborsClassifier(3, algorithm='kd_tree', n_jobs=-1)
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)

print(f"test score is {f1_score(Y_test, y_pred, average='micro')}")