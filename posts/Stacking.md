---
layout: post
title: Stacked Generalization
lang: pt
date: 2019-04-22 12:59:07
tags: [ensemble, meta-learning]
author: Jader Martins
comments: true
---

Por ser uma tecnica relativamente nova, o "Stacked Generalization" ainda não tem uma tradução amplamente adotada, aqui proponho "empilhamento", tradução literal para seu outro nome (stacking), como o termo para me referir a ela. Introduzida por Wolpert em 1992[^1], essa tecnica de generalização consiste em combinar de formar não linear estimadores para corrigir seus vieses a um dado conjunto de treino, agregando suas capacidades para que se tenha uma melhor previsão.

<img src="/images/stacking.png" width=600px>

Em uma [postagem anterior](/posts/Linear-Ensemble.html) apresentei a combinação linear de estimadores, nela ajustamos $N$ modelos a um conjunto de dados \\(D\\) e a priori definimos pesos \\(W\\) para eles combinando em um somatório:


$$\text{given a priori} \ W = (w_1,w_2,...w_N) \ \text{and} \sum W = 1$$

$$\sum_{i=1}^{N} w_{i}M_{i}$$

com isso a media ponderada das predições no geral vão ser menos enviesadas para certas regiões e podem generalizar mais, porém este metodo apresenta duas limitações, os pesos não podem ser alterados depois de verificar o desempenho (se não estariamos agindo como um meta-estimador em cima dos dados de teste) e é uma combinação extremamente simples, não aproveitando bem os pontos fortes dos estimadores \\(M\\) para certas regiões.

Wolpert então propõe uma alternativa a isso, e se tornasemos os pesos \\(W\\) em um problema de aprendizado? ou melhor, não só aprendessemos como combinar nossas predições mas também as combinassemos de forma não-linear usando um meta-estimador?

Meta-estimadores são aqueles que usam modelos base para combina-los ou seleciona-los para melhorar em uma metrica de desempenho, por exemplo você leitor quando decide entre usar uma random-forest ou uma regressão logistica para prever o seu modelo você está sendo um meta-estimador. Porém aqui surge o problema de generalização, se continuar melhorando sua regressão ou rforest você poderá acabar dando overfitting aos dados e não conseguindo generalizar, aqui então é necessário aplicar tecnicas de validação cruzada para selecionar o modelo, o mesmo ocorrerá para o empilhamento.

Para o empilhamento é ideal que o dataset seja relativamente grande, o conselho do autor é pelo menos mil registros. Começamos nosso exemplo carregando um dataset relativamente grande, 20mil registros, esse dataset tem como atributos caracteristicas de casas da california e como valor alvo o preço dela, os dados já estão normalizados e não iremos fazer qualquer alteração nele.


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
      <th>0</th>
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
      <th>1</th>
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
      <th>2</th>
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
      <th>3</th>
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
      <th>4</th>
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



Aqui separamos em treino e teste de forma (pseudo)aleatorizada para no final avaliarmos o desempenho.


```python
xtrain, xtest, ytrain, ytest = train_test_split(df.drop('Price', axis=1), df.Price, test_size=.3, random_state=42)
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
      <th>7061</th>
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
      <th>14689</th>
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
      <th>17323</th>
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
      <th>10056</th>
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
      <th>15750</th>
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



Agora carregamos a validação-cruzada especificamente a [KFold](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold), para que não "percamos" muitos dados, e os modelos que serão usados, aqui não há uma regra de dedo sobre os modelos base, fica a seu criterio, porém para o meta-estimador é usualmente aplicado boosting trees. Aqui escolhi arbitrariamente [kNN](https://scikit-learn.org/stable/modules/neighbors.html#nearest-neighbors-regression) e [ElasticNet](https://scikit-learn.org/stable/modules/linear_model.html#elastic-net), mas como meta-estimador usarei o [xgboost](https://xgboost.readthedocs.io/en/latest/tutorials/model.html).


```python
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

import xgboost as xgb

en = ElasticNet()
knn = KNeighborsRegressor()
gbm = xgb.XGBRegressor(n_estimators=1000) # we will early stop to not overfit
```

Agora se inicia a criação dos atributos empilhados, para garantir que não tenha vieses e não fiquemos com poucos dados para treinar o meta-etimador os criamos por kfolds, sendo gerados os subconjuntos treino e teste, treinamos o modelo no conjunto de treino e predizemos o valor para o conjunto de teste, da seguinte forma:


```python
kf = KFold(20, shuffle=True)
xtrain['en'] = 0
for train_index, test_index in kf.split(xtrain):
    en.fit(xtrain.iloc[train_index, :-1], ytrain.iloc[train_index])
    xtrain.iloc[test_index,8] = en.predict(xtrain.iloc[test_index, :-1])
```

Fazemos o mesmo para o outro modelo.


```python
kf = KFold(20, shuffle=True)
xtrain['knn'] = 0

for train_index, test_index in kf.split(xtrain):
    knn.fit(xtrain.iloc[train_index, :-2], ytrain.iloc[train_index])
    xtrain.iloc[test_index,9] = knn.predict(xtrain.iloc[test_index, :-2])
```


```python
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
      <th>7061</th>
      <td>4.1312</td>
      <td>35.0</td>
      <td>5.882353</td>
      <td>0.975490</td>
      <td>1218.0</td>
      <td>2.985294</td>
      <td>33.93</td>
      <td>-118.02</td>
      <td>2.203225</td>
      <td>2.108000</td>
    </tr>
    <tr>
      <th>14689</th>
      <td>2.8631</td>
      <td>20.0</td>
      <td>4.401210</td>
      <td>1.076613</td>
      <td>999.0</td>
      <td>2.014113</td>
      <td>32.79</td>
      <td>-117.09</td>
      <td>1.706363</td>
      <td>1.973400</td>
    </tr>
    <tr>
      <th>17323</th>
      <td>4.2026</td>
      <td>24.0</td>
      <td>5.617544</td>
      <td>0.989474</td>
      <td>731.0</td>
      <td>2.564912</td>
      <td>34.59</td>
      <td>-120.14</td>
      <td>2.091721</td>
      <td>2.197800</td>
    </tr>
    <tr>
      <th>10056</th>
      <td>3.1094</td>
      <td>14.0</td>
      <td>5.869565</td>
      <td>1.094203</td>
      <td>302.0</td>
      <td>2.188406</td>
      <td>39.26</td>
      <td>-121.00</td>
      <td>1.698103</td>
      <td>2.160600</td>
    </tr>
    <tr>
      <th>15750</th>
      <td>3.3068</td>
      <td>52.0</td>
      <td>4.801205</td>
      <td>1.066265</td>
      <td>1526.0</td>
      <td>2.298193</td>
      <td>37.77</td>
      <td>-122.45</td>
      <td>2.195922</td>
      <td>2.388002</td>
    </tr>
  </tbody>
</table>
</div>



Agora que criamos as features vamos avaliar os modelos nos dados brutos, sem as features empilhadas para verificar seus desempenhos:


```python
from sklearn.metrics import mean_squared_error
en.fit(xtrain.iloc[:,:-2], ytrain)
ypred_en = en.predict(xtest)
print(mean_squared_error(ytest, ypred_en))

knn.fit(xtrain.iloc[:,:-2], ytrain)
ypred_knn = knn.predict(xtest)
print(mean_squared_error(ytest, ypred_knn))
```

    0.7562926012142394
    1.136942049088978


Agora criamos as features com os modelos treinados para os dados de teste:


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
      <th>20046</th>
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
      <th>3024</th>
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
      <th>15663</th>
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
      <th>20484</th>
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
      <th>9814</th>
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



Com os atributos empilhados em mãos agora treinamos dois modelos, um sem utiliza-los, para efeito de comparação e outro utilizando, vamos comparar os resultados:


```python
#Without stacked features
gbm.fit(xtrain.iloc[:,:-2], ytrain.values,
        eval_set=[(xtest.iloc[:,:-2],ytest.values)],
        early_stopping_rounds=10,
        verbose=False)
ypred = gbm.predict(xtest.iloc[:,:-2])
print("Without stacked features", mean_squared_error(ytest, ypred))

# With stacked features
gbm.fit(xtrain, ytrain.values,
        eval_set=[(xtest,ytest.values)],
        early_stopping_rounds=10,
        verbose=False)
ypred = gbm.predict(xtest)
print("With stacked features", mean_squared_error(ytest, ypred))
```

    Without stacked features 0.23318853095188977
    With stacked features 0.22410766547017014


Nos tivemos uma melhora significativa usando atributos "empilhados", concluindo nosso meta-estimador aprende a melhor forma de combinar as features de outros estimadores, aprendendo seus erros de generalização e como os corrigir, garantindo uma generalização muito melhor.

#### References

[Stacked Generalization](http://machine-learning.martinsewell.com/ensembles/stacking/)

[^1]: https://www.sciencedirect.com/science/article/pii/S0893608005800231
