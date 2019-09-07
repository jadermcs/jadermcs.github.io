---
layout: post
title: Stacked Generalization
lang: pt
date: 2019-04-22 12:59:07
tags: [ensemble, meta-learning]
author: Jader Martins
---

Introduced by Wolpert in 1992[^1], this generalization technique consists of combining nonlinear estimators to correct their biases to a given training set, adding their capabilities for better prediction[^2].

<img src="/images/stacking.png" width=600px>

In a [previous post](/posts/Linear-Ensemble.html) I presented the linear combination of estimators, in it we adjusted $N$ models to a $D$ dataset and a priori we defined $W$ weights for them by combining into one summation:

$$\sum_{i=1}^{N} w_{i}M_{i}$$

$$\text{given a priori} \ W = (w_1,w_2,...w_N) \ \text{and} \sum W = 1$$

With this the weighted average of the predictions in general will be less biased for certain regions and may generalize more, but this method has two limitations, the weights cannot be changed after verifying the performance (if we would not be acting as a meta-estimator in test data) and is an extremely simple combination, not leveraging the strengths of the $M$ estimators for certain regions.

Wolpert then proposes an alternative to this, what if we make $W$ pesos a learning problem? or rather, not only learn how to combine our predictions but also combine them nonlinearly using a meta estimator?

Meta estimators are those who use base models to combine them or select them to improve on a performance metric, for example you reader when deciding between using a random forest or a logistic regression to predict your model you are being a meta estimator. But here the problem of generalization arises, if you continue to improve your regression or rforest you may end up overfitting the data and not being able to generalize, here then it is necessary to apply cross validation techniques to select the model, the same will happen for the stacking.

For stacking it is ideal that the dataset is relatively large, the author's advice is at least one thousand records. We start our example by loading a relatively large dataset, 20,000 records, this dataset has as characteristic attributes of california houses and as a target value its price, the data is already normalized and we will not make any changes to it.


```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import pandas as pd

dataset = fetch_california_housing()
df = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
df['Price'] = dataset.target
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>Price</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>8.3252</td>
      <td>41.0</td>
      <td>6.984127</td>
      <td>1.023810</td>
      <td>322.0</td>
      <td>2.555556</td>
      <td>37.88</td>
      <td>-122.23</td>
      <td>4.526</td>
    </tr>
    <tr>
      <td>1</td>
      <td>8.3014</td>
      <td>21.0</td>
      <td>6.238137</td>
      <td>0.971880</td>
      <td>2401.0</td>
      <td>2.109842</td>
      <td>37.86</td>
      <td>-122.22</td>
      <td>3.585</td>
    </tr>
    <tr>
      <td>2</td>
      <td>7.2574</td>
      <td>52.0</td>
      <td>8.288136</td>
      <td>1.073446</td>
      <td>496.0</td>
      <td>2.802260</td>
      <td>37.85</td>
      <td>-122.24</td>
      <td>3.521</td>
    </tr>
    <tr>
      <td>3</td>
      <td>5.6431</td>
      <td>52.0</td>
      <td>5.817352</td>
      <td>1.073059</td>
      <td>558.0</td>
      <td>2.547945</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.413</td>
    </tr>
    <tr>
      <td>4</td>
      <td>3.8462</td>
      <td>52.0</td>
      <td>6.281853</td>
      <td>1.081081</td>
      <td>565.0</td>
      <td>2.181467</td>
      <td>37.85</td>
      <td>-122.25</td>
      <td>3.422</td>
    </tr>
  </tbody>
</table>
</div>



Here we separate into training and testing in a (pseudo) random way to finally evaluate performance.


```python
xtrain, xtest, ytrain, ytest =\
    train_test_split(df.drop('Price', axis=1), df.Price, test_size=.3,
                     random_state=42)
xtrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7061</td>
      <td>4.1312</td>
      <td>35.0</td>
      <td>5.882353</td>
      <td>0.975490</td>
      <td>1218.0</td>
      <td>2.985294</td>
      <td>33.93</td>
      <td>-118.02</td>
    </tr>
    <tr>
      <td>14689</td>
      <td>2.8631</td>
      <td>20.0</td>
      <td>4.401210</td>
      <td>1.076613</td>
      <td>999.0</td>
      <td>2.014113</td>
      <td>32.79</td>
      <td>-117.09</td>
    </tr>
    <tr>
      <td>17323</td>
      <td>4.2026</td>
      <td>24.0</td>
      <td>5.617544</td>
      <td>0.989474</td>
      <td>731.0</td>
      <td>2.564912</td>
      <td>34.59</td>
      <td>-120.14</td>
    </tr>
    <tr>
      <td>10056</td>
      <td>3.1094</td>
      <td>14.0</td>
      <td>5.869565</td>
      <td>1.094203</td>
      <td>302.0</td>
      <td>2.188406</td>
      <td>39.26</td>
      <td>-121.00</td>
    </tr>
    <tr>
      <td>15750</td>
      <td>3.3068</td>
      <td>52.0</td>
      <td>4.801205</td>
      <td>1.066265</td>
      <td>1526.0</td>
      <td>2.298193</td>
      <td>37.77</td>
      <td>-122.45</td>
    </tr>
  </tbody>
</table>
</div>



We now load cross-validation specifically into [KFold](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold) so that we don't "lose" a lot of data, and the templates that will be used , here there is no rule of thumb about the base models, it is up to you, but for the meta-estimator is usually applied boosting trees. Here I arbitrarily chose [kNN](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression) and [ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net), but as a meta-estimator I will use [xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html).


```python
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

from xgboost import XGBRFRegressor

en = ElasticNet()
knn = KNeighborsRegressor()
# we will early stop to not overfit
gbm = XGBRFRegressor(n_jobs=-1, objective='reg:squarederror')
```

Now the creation of the stacked attributes begins, to make sure that there are no biases and we don't have little data to train the meta-estimator we create them by kfolds, being generated the training and test subsets, we train the model in the training set and we predict the value for the test set as follows:


```python
kf = KFold(20, shuffle=True)
xtrain['en'] = 0
for train_index, test_index in kf.split(xtrain):
    en.fit(xtrain.iloc[train_index, :-1], ytrain.iloc[train_index])
    xtrain.iloc[test_index,8] = en.predict(xtrain.iloc[test_index, :-1])
```

We do the same for the other model.


```python
kf = KFold(20, shuffle=True)
xtrain['knn'] = 0

for train_index, test_index in kf.split(xtrain):
    knn.fit(xtrain.iloc[train_index, :-2], ytrain.iloc[train_index])
    xtrain.iloc[test_index,9] = knn.predict(xtrain.iloc[test_index, :-2])

xtrain.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>en</th>
      <th>knn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>7061</td>
      <td>4.1312</td>
      <td>35.0</td>
      <td>5.882353</td>
      <td>0.975490</td>
      <td>1218.0</td>
      <td>2.985294</td>
      <td>33.93</td>
      <td>-118.02</td>
      <td>2.208931</td>
      <td>2.108000</td>
    </tr>
    <tr>
      <td>14689</td>
      <td>2.8631</td>
      <td>20.0</td>
      <td>4.401210</td>
      <td>1.076613</td>
      <td>999.0</td>
      <td>2.014113</td>
      <td>32.79</td>
      <td>-117.09</td>
      <td>1.705684</td>
      <td>1.809200</td>
    </tr>
    <tr>
      <td>17323</td>
      <td>4.2026</td>
      <td>24.0</td>
      <td>5.617544</td>
      <td>0.989474</td>
      <td>731.0</td>
      <td>2.564912</td>
      <td>34.59</td>
      <td>-120.14</td>
      <td>2.098392</td>
      <td>1.683200</td>
    </tr>
    <tr>
      <td>10056</td>
      <td>3.1094</td>
      <td>14.0</td>
      <td>5.869565</td>
      <td>1.094203</td>
      <td>302.0</td>
      <td>2.188406</td>
      <td>39.26</td>
      <td>-121.00</td>
      <td>1.694140</td>
      <td>1.792000</td>
    </tr>
    <tr>
      <td>15750</td>
      <td>3.3068</td>
      <td>52.0</td>
      <td>4.801205</td>
      <td>1.066265</td>
      <td>1526.0</td>
      <td>2.298193</td>
      <td>37.77</td>
      <td>-122.45</td>
      <td>2.194403</td>
      <td>2.388002</td>
    </tr>
  </tbody>
</table>
</div>



Now that we have created the features let's evaluate the models in the raw data without the stacked features to check their performances:


```python
from sklearn.metrics import mean_squared_error
en.fit(xtrain.iloc[:,:-2], ytrain)
ypred_en = en.predict(xtest)
print(mean_squared_error(ytest, ypred_en))

knn.fit(xtrain.iloc[:,:-2], ytrain)
ypred_knn = knn.predict(xtest)
print(mean_squared_error(ytest, ypred_knn))
```

    0.7562926012142382
    1.136942049088978


Now we create the features with the trained models for the test data:


```python
xtest['en'] = ypred_en
xtest['knn'] = ypred_knn
xtest.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MedInc</th>
      <th>HouseAge</th>
      <th>AveRooms</th>
      <th>AveBedrms</th>
      <th>Population</th>
      <th>AveOccup</th>
      <th>Latitude</th>
      <th>Longitude</th>
      <th>en</th>
      <th>knn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>20046</td>
      <td>1.6812</td>
      <td>25.0</td>
      <td>4.192201</td>
      <td>1.022284</td>
      <td>1392.0</td>
      <td>3.877437</td>
      <td>36.06</td>
      <td>-119.01</td>
      <td>1.470084</td>
      <td>1.6230</td>
    </tr>
    <tr>
      <td>3024</td>
      <td>2.5313</td>
      <td>30.0</td>
      <td>5.039384</td>
      <td>1.193493</td>
      <td>1565.0</td>
      <td>2.679795</td>
      <td>35.14</td>
      <td>-119.46</td>
      <td>1.744788</td>
      <td>1.0822</td>
    </tr>
    <tr>
      <td>15663</td>
      <td>3.4801</td>
      <td>52.0</td>
      <td>3.977155</td>
      <td>1.185877</td>
      <td>1310.0</td>
      <td>1.360332</td>
      <td>37.80</td>
      <td>-122.44</td>
      <td>2.233643</td>
      <td>2.8924</td>
    </tr>
    <tr>
      <td>20484</td>
      <td>5.7376</td>
      <td>17.0</td>
      <td>6.163636</td>
      <td>1.020202</td>
      <td>1705.0</td>
      <td>3.444444</td>
      <td>34.28</td>
      <td>-118.72</td>
      <td>2.413336</td>
      <td>2.2456</td>
    </tr>
    <tr>
      <td>9814</td>
      <td>3.7250</td>
      <td>34.0</td>
      <td>5.492991</td>
      <td>1.028037</td>
      <td>1063.0</td>
      <td>2.483645</td>
      <td>36.62</td>
      <td>-121.93</td>
      <td>2.088660</td>
      <td>1.6690</td>
    </tr>
  </tbody>
</table>
</div>



With the stacked attributes in hand now we train two models, one without using them, for comparison and another using, let's compare the results:


```python
#Without stacked features
gbm.fit(xtrain.iloc[:,:-2], ytrain.values,
        eval_set=[(xtest.iloc[:,:-2],ytest.values)],
        early_stopping_rounds=20,
        verbose=False)
ypred = gbm.predict(xtest.iloc[:,:-2])
print("Without stacked features", mean_squared_error(ytest, ypred))
# With stacked features
gbm.fit(xtrain, ytrain.values,
        eval_set=[(xtest,ytest.values)],
        early_stopping_rounds=20,
        verbose=False)
ypred = gbm.predict(xtest)
print("With stacked features", mean_squared_error(ytest, ypred))
```

    Without stacked features 0.5828429815199971
    With stacked features 0.5359477372727965


We've had a significant improvement using "stacked" attributes, concluding our meta estimator learns the best way to combine the features of other estimators, learning their generalization errors and how to correct them, ensuring a much better generalization.

#### References

[Stacked Generalization](http://machine-learning.martinsewell.com/ensembles/stacking/)

[^1]: https://www.sciencedirect.com/science/article/pii/S0893608005800231

[^2]: HASTIE, Trevor et al. The elements of statistical learning: data mining, inference and prediction. P. 252, 2005
