# WordsClusterBySynonyms
Words clustering using synonyms 

This class is able to create clusters by using the definition of synonyms inside NLTK. Let's see an example.

```python
import pandas as pd
import WordsClusterBySynonyms as wcbs
```

In this case we decided to use a list of italian verbs.

```python
verbs = [
    'cogliere', 'intagliare', 'ragguagliare', 'dilazionare', 'tuffare',
    'dissipare', 'indisporre', 'complottare', 'contraddire', 'sconoscere',
    'sgocciolare', 'ridimensionare', 'ammansire', 'stuzzicare', 'rintuzzare',
    ...
    'autenticare', 'programmare', 'assassinare', 'immalinconire', 'esalare',
    'istigare', 'abiurare', 'curare', 'tranciare', 'tracciare', 'vagolare',
    'raddolcire', 'sfinire', 'confrontare', 'indispettire','fare','avere','vivere'
]
```

```python
verbs = pd.DataFrame(verbs)
verbs.columns = ['verbs']
```
WordsClusterBySynonyms requires a dataframe in which you have to specify the name of the target column and the language obviously.

The first function inside **WordClusterBySynonyms** is **get_synonyms_pandas**. It applies on the dataframe the generation of synonyms by creating a new columns.

```python
wc = wcbs.WordsClusterBySynonyms(verbs, 'verbs', lang='ita')
df = wc.get_synonyms_pandas()
```


```python
wc.plot_hist(df)
```
[![hist_all.jpg](https://s17.postimg.org/ngd79azu7/hist_all.jpg)](https://postimg.org/image/5dk4i33zf/)

Using set_threshold you can repeat **get_synonyms_pandas** with a threshold

```python
df = wc.set_threshold(20, df)
```
Using **plot_hist** you can check if in your list of words there are words with associate a huge number of synonyms. These words are a problem, because they tend to create few huge clusters with our definition of distance.

```python
wc.plot_hist(df)
```
[![hist_no_higher.jpg](https://s17.postimg.org/5qbioa1ov/hist_no_higher.jpg)](https://postimg.org/image/e8kysm87f/)

#### DISTANCE
Given two different words (A and B) with associated two lists of synonyms ( ![sa](https://latex.codecogs.com/gif.latex?S_A)  and ![sb](https://latex.codecogs.com/gif.latex?S_B)). A is equal to B if ![sa](https://latex.codecogs.com/gif.latex?S_A) is equal to ![sb](https://latex.codecogs.com/gif.latex?S_B). A is totally different from B if there is an empty intersection between ![sa](https://latex.codecogs.com/gif.latex?S_A) and ![sb](https://latex.codecogs.com/gif.latex?S_B).

The formula we used is:

![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BS_A%20%5Ccap%20S_B%7D%7Bmin%28len%28S_A%29%2Clen%28S_B%29%29%7D)

You can choose between min or max, or if you would like to use your definition of distance:

```python
    def mydistance_name():
        ...
        return ...

    wc.create_distance_matrix(mydistance= mydistance_name, criteria=None, verbose=True)
```
```python
matrix = wc.create_distance_matrix(criteria=min, verbose=True)
wc.plot_eps_ncluster(matrix, ntot=10, min_samples=6)
```
[![plot_eps_clusters.jpg](https://s17.postimg.org/g6n1cmqof/plot_eps_clusters.jpg)](https://postimg.org/image/bkqx4a557/)
[![plot_eps_not_clustered.jpg](https://s17.postimg.org/oowhgzkcf/plot_eps_not_clustered.jpg)](https://postimg.org/image/9sxy9e8xn/)

The function **run_cluster** uses the DBSCAN implemented in **sklearn**. 
You can find the documentation here: http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html

```python
result = wc.run_cluster(0.3,6, matrix)
```

```python
wc.plot_cluster_k(matrix, 'contraddire')
```
[![contraddire.jpg](https://s17.postimg.org/t9inwcbvz/contraddire.jpg)](https://postimg.org/image/fslpdh1kb/)
