
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import learning_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import make_scorer ,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
from keras import models
from keras import layers
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import svm


data = pd.read_csv('dataset/cc-80.csv')[:2000]
data_test = pd.read_csv('dataset/cc-20.csv')[:400]
print('Train Data Shape =',data.shape)
print('Columns =',data.columns)

data.describe()

data.head()

data['Target']=LabelEncoder().fit_transform(data['Target'])
corr = data.corr()
corr['Target'].sort_values()

features = [ 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 
'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20''V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31']
y=data['Target']
X=data[features]

y_test=data_test['Target']
X_test=data_test[features]

s = (X.dtypes == 'object')
object_cols = list(s[s].index)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols = pd.DataFrame(OH_encoder.fit_transform(X[object_cols]))

OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[object_cols]))

OH_cols.index = X.index
OH_cols_test.index = X_test.index

num_X = X.drop(object_cols, axis=1)
num_X_test = X_test.drop(object_cols, axis=1)

X = pd.concat([num_X, OH_cols], axis=1)
X_test = pd.concat([num_X_test, OH_cols_test], axis=1)

X.columns=X.columns.astype(str)
X_test.columns=X.columns.astype(str)


print('input size = ',X.shape)

def plot_grid_search(cv_results, name_param):    
    params=[d[name_param] for d in cv_results['params']]
    scores=cv_results['mean_test_score']
    fig = plt.figure()
    plt.plot(params,scores)
    fig.suptitle('Grid Search Scores', fontsize=20)
    plt.xlabel(name_param)
    plt.ylabel('CV Average Score')
    plt.grid('on')
def plot_learning_curve(estimator, X, y,train_sizes=np.linspace(0.01, 1.0, 10)):
        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator,X,y,cv=5,n_jobs=-1,train_sizes=train_sizes,scoring = make_scorer(accuracy_score),return_times=True)
        fit_times_means = np.mean(fit_times, axis =1)
        print('Fitting time (seconds) =' , fit_times_means[-1])
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        fig = plt.figure()
        plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
        plt.plot(train_sizes, test_scores_mean, "o-", color="b", label="Cross-validation score")
        fig.suptitle('DS Salary Learning curves', fontsize=20)
        plt.xlabel('Training size')
        plt.ylabel('Accuracy')
        plt.legend(loc="best")
        plt.grid('on')

DT = DecisionTreeClassifier(random_state=7)

n_scores = cross_val_score(DT, X, y, scoring='accuracy', cv=6, n_jobs=-1)

DT.fit(X,y)

pred_y=DT.predict(X_test)

plot_learning_curve(DT, X, y)

ccp_alphas = DT.cost_complexity_pruning_path(X, y)["ccp_alphas"]
ccp_alpha_grid_search = GridSearchCV(estimator = DT,
                                    scoring = make_scorer(accuracy_score),
                                    param_grid=ParameterGrid({"ccp_alpha":[[alpha] for alpha in ccp_alphas]}),
                                    cv=5,
                                    n_jobs = -1)

ccp_alpha_grid_search.fit(X, y)

DT_prunned=ccp_alpha_grid_search.best_estimator_

n_scores = cross_val_score(DT_prunned, X, y, scoring='accuracy', cv=5, n_jobs=-1)

DT_prunned.fit(X,y)

pred_y=DT_prunned.predict(X_test)

input_shape = [X.shape[1]]

def create_network():
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=64, activation='relu', input_shape=input_shape))

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=64, activation='relu'))
    
    network.add(layers.Dense(units=64, activation='relu'))

    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='sigmoid'))

    # Compile neural network
    network.compile(loss='binary_crossentropy', # Cross-entropy
                    optimizer='adam', # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network

Neural_net = KerasClassifier(build_fn=create_network, 
                                 epochs=100, 
                                 batch_size=1000, 
                                 verbose=0)

n_scores = cross_val_score(Neural_net, X, y, scoring='accuracy', cv=5, n_jobs=-1)

plot_learning_curve(Neural_net, X, y)

Neural_net.fit(X,y)

pred_y=Neural_net.predict(X_test)

ada_boost = AdaBoostClassifier(base_estimator=DT_prunned,learning_rate=1,random_state=7)
# evaluate the model
n_scores = cross_val_score(ada_boost, X, y, scoring='accuracy', cv=5, n_jobs=-1)

ada_boost.fit(X,y)

pred_y=ada_boost.predict(X_test)

plot_learning_curve(ada_boost, X, y)

lr_grid_search = GridSearchCV(estimator = ada_boost,
                    scoring = make_scorer(accuracy_score),
                    param_grid=ParameterGrid({"learning_rate":[[10**-k] for k in range(0,10)]}),
                             cv=5,
                    n_jobs=-1)     

lr_grid_search.fit(X, y)

ada_optimal=lr_grid_search.best_estimator_
print('Best parameter =',lr_grid_search.best_params_)
n_scores = cross_val_score(ada_optimal, X, y, scoring='accuracy', cv=5, n_jobs=-1)

ada_optimal.fit(X,y)

pred_y=ada_optimal.predict(X_test)
plot_learning_curve(ada_optimal, X, y)

plot_grid_search(lr_grid_search.cv_results_,'learning_rate')

svm1 = svm.SVC() 

n_scores = cross_val_score(svm1, X, y, scoring='accuracy', cv=5, n_jobs=-1)

svm1.fit(X,y)

pred_y=svm1.predict(X_test)
plot_learning_curve(svm1, X, y)

svm2 = svm.SVC(kernel='poly') 

n_scores = cross_val_score(svm2, X, y, scoring='accuracy', cv=5, n_jobs=-1)

svm2.fit(X,y)

pred_y=svm2.predict(X_test)
plot_learning_curve(svm2, X, y)


degree_grid_search = GridSearchCV(estimator = svm2,
                    scoring = make_scorer(accuracy_score),
                    param_grid=ParameterGrid({"degree":[[k] for k in range(1,25)]}),
                             cv=5,
                    n_jobs=-1)     

degree_grid_search.fit(X, y)

svm2_optimal=degree_grid_search.best_estimator_
n_scores = cross_val_score(svm2_optimal, X, y, scoring='accuracy', cv=5, n_jobs=-1)

svm2_optimal.fit(X,y)

pred_y=svm2_optimal.predict(X_test)

plot_learning_curve(svm2_optimal, X, y)

plot_grid_search(degree_grid_search.cv_results_,'degree')

knn = KNeighborsClassifier(n_neighbors=3)
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(knn, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

knn.fit(X,y)

pred_y=knn.predict(X_test)
plot_learning_curve(knn, X, y)


k_grid_search = GridSearchCV(estimator = knn,
                    scoring = make_scorer(accuracy_score),
                    param_grid=ParameterGrid({"n_neighbors":[[k] for k in range(1,31)]}),
                             cv=5,
                    n_jobs=-1)        

k_grid_search.fit(X, y)

knn_optimal = k_grid_search.best_estimator_

n_scores = cross_val_score(knn_optimal, X, y, scoring='accuracy', cv=5, n_jobs=-1)
knn_optimal.fit(X,y)
pred_y=knn_optimal.predict(X_test)
plot_learning_curve(knn_optimal, X, y)
plot_grid_search(k_grid_search.cv_results_,'n_neighbors')