# WordsClusterBySynonyms
Words clustering using synonyms 

This class is able to create cluster by using the definition of synonyms inside NLTK. Let's see an example.

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
WordsClusterBySynonyms requires a dataframe in which you have to specify the name of the target column and the languages obviously.

The first function inside **WordClusterBySynonyms** is **get_synonyms_pandas**. It applies on the dataframe the generation of synonyms by creating a new columns.

```python
wc = wcbs.WordsClusterBySynonyms(verbs, 'verbs', lang='ita')
df = wc.get_synonyms_pandas()
```


```python
wc.plot_hist(df)
```
[![hist_all.jpg](https://s17.postimg.org/93f5wwqcv/hist_all.jpg)](https://postimg.org/image/tnjzve63v/)

Using set_treshold you can repeat **get_synonyms_pandas** with a threshold

```python
df = wc.set_treshold(20, df)
```
Using **plot_hist** you can check if in your list of words there are words with associate a huge number of synonyms. There words are a problem, because of our definition of distance they tend to create few huge cluster.

```python
wc.plot_hist(df)
```
[![hist_no_higher.jpg](https://s17.postimg.org/vs4cwii1b/hist_no_higher.jpg)](https://postimg.org/image/dcjvz43wr/)
[![sa](https://latex.codecogs.com/gif.latex?S_A)

#### DISTANCE
Given two different words (A and B) with associated two lists of synonyms ([![sa](https://latex.codecogs.com/gif.latex?S_A)  and $$S_B$$). A is equal to B if $S_A$ is equal to $S_B$. A is totally different from B if there is an empty intersection between $$S_A$$ and $$S_B$$.

### scriverlo in formula

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

```python
result = wc.run_cluster(0.3,6, matrix)
```

```python
wc.plot_cluster_k(matrix, 'contraddire')
```
[![contraddire.jpg](https://s17.postimg.org/t9inwcbvz/contraddire.jpg)](https://postimg.org/image/fslpdh1kb/)
