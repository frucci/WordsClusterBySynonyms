# WordsClusterBySynonyms
Words clustering using synonyms 

```python
import pandas as pd
import WordsClusterBySynonyms as wcbs
```

```python
verbs = [
    'cogliere', 'intagliare', 'ragguagliare', 'dilazionare', 'tuffare',
    'dissipare', 'indisporre', 'complottare', 'contraddire', 'sconoscere',
    'sgocciolare', 'ridimensionare', 'ammansire', 'stuzzicare', 'rintuzzare',
    'stropicciare', 'funestare', 'scuotere', 'suffragare', 'scomodare',
    'animare', 'distrarre', 'spingere', 'allettare', 'sussurrare', 'spossare',
    'smentire', 'terrorizzare', 'delineare', 'movimentare', 'bofonchiare',
    'vivacizzare', 'paragonare', 'deprecare', 'abbacchiare', 'scrollare',
    'smussare', 'spargere', 'stremare', 'scatenare', 'accreditare', 'spalmare',
    'contribuire', 'addobbare', 'sbalordire', 'affettare', 'terrificare',
    'intimorire', 'errare', 'protestare', 'rimproverare', 'accompagnare',
    'accentrare', 'innamorare', 'riordinare', 'massacrare', 'pellegrinare',
    'avvolgere', 'dondolare', 'rattristare', 'offuscare', 'fregiare',
    'sorreggere', 'circondare', 'bruttare', 'concludere', 'importunare',
    'dilettare', 'brigare', 'frustare', 'manipolare', 'confutare', 'stupire',
    'conseguire', 'angariare', 'sollazzare', 'incidere', 'ammollire',
    'corrugare', 'blandire', 'comprovare', 'adulterare', 'stillare',
    'spiegazzare', 'dolcificare', 'portare', 'abbandonare', 'ordire',
    'rammaricare', 'evocare', 'rimandare', 'ferire', 'confortare', 'cessare',
    'condannare', 'ritardare', 'bacchettare', 'scortare', 'orientare',
    'incoraggiare', 'infangare', 'tardare', 'ricreare', 'rinfrancare',
    'viziare', 'guidare', 'desolare', 'certificare', 'ammonire', 'danneggiare',
    'vagabondare', 'intingere', 'differire', 'gestire', 'innescare',
    'lamentare', 'vulnerare', 'ristrutturare', 'assestare', 'deprimere',
    'lasciare', 'tartagliare', 'vidimare', 'intrallazzare', 'nuocere',
    'scannare', 'dilapidare', 'impappinare', 'alterare', 'sedare', 'avvincere',
    'inorridire', 'impuntare', 'mormorare', 'concretare', 'peregrinare',
    'infirmare', 'incantare', 'sgualcire', 'cingere', 'persuadere', 'sgridare',
    'insudiciare', 'frustrare', 'sublimare', 'intronare', 'innervosire',
    'ritrattare', 'sprizzare', 'polarizzare', 'straccare', 'fomentare',
    'sperdere', 'rabbuffare', 'trascurare', 'insignire', 'tessere', 'eccitare',
    'stordire', 'infamare', 'risanare', 'sboccare', 'inviare', 'avvilire',
    'avvalorare', 'testimoniare', 'devastare', 'posporre', 'annichilire',
    'dimenare', 'calmare', 'spintonare', 'schizzare', 'umettare', 'prostrare',
    'contattare', 'svagare', 'trasferire', 'invalidare', 'tediare',
    'accerchiare', 'ledere', 'ossessionare', 'sgozzare', 'logorare', 'spedire',
    'menare', 'intrigare', 'alienare', 'sporcare', 'intessere', 'illuminare',
    'snobbare', 'sparlare', 'guastare', 'redarguire', 'solleticare',
    'trucidare', 'procrastinare', 'invogliare', 'soddisfare', 'mortificare',
    'saziare', 'contentare', 'cospirare', 'ignorare', 'raggrinzire',
    'incalzare', 'parlottare', 'macellare', 'congiurare', 'scolpire',
    'tramare', 'castrare', 'abbattere', 'appannare', 'rampognare',
    'demoralizzare', 'fustigare', 'mugugnare', 'diffamare', 'abbozzare',
    'dipartire', 'scolare', 'abbellire', 'apostatare', 'inviluppare',
    'ispirare', 'defezionare', 'bagnare', 'sanare', 'stregare',
    'sovrintendere', 'rischiarare', 'deplorare', 'deteriorare', 'alleviare',
    'tranquillizzare', 'mitigare', 'maneggiare', 'incanalare', 'contaminare',
    'calunniare', 'affaticare', 'conquistare', 'ornare', 'smettere',
    'gocciolare', 'meravigliare', 'imbrattare', 'tralasciare', 'spaventare',
    'seminare', 'pilotare', 'divorare', 'grugnire', 'brontolare',
    'riscontrare', 'sconfessare', 'ammazzare', 'aizzare', 'ravvolgere',
    'consumare', 'interrompere', 'adornare', 'attestare', 'guadagnare',
    'rimbrottare', 'inoltrare', 'deliziare', 'prorogare', 'disconoscere',
    'terminare', 'smontare', 'sperperare', 'promuovere', 'strabiliare',
    'zampillare', 'screditare', 'razziare', 'pacare', 'biasimare', 'buttare',
    'sofisticare', 'confezionare', 'estraniare', 'strapazzare', 'disastrare',
    'sciupare', 'avvisare', 'manovrare', 'ciangottare', 'biascicare',
    'confermare', 'magnetizzare', 'promulgare', 'stupefare', 'spregiare',
    'scaturire', 'riorganizzare', 'rinviare', 'punzecchiare', 'ristorare',
    'vellicare', 'scordare', 'lacrimare', 'raggrinzare', 'condurre',
    'disonorare', 'falsare', 'incomodare', 'posticipare', 'depredare',
    'macchinare', 'compromettere', 'impastare', 'sommuovere', 'canalizzare',
    'partire', 'molcere', 'imbucare', 'inumidire', 'divagare', 'inficiare',
    'uscire', 'esaudire', 'profilare', 'convalidare', 'ondeggiare',
    'prodigare', 'affatturare', 'insozzare', 'spendere', 'sprigionare',
    'addomesticare', 'scoraggiare', 'macchiare', 'spolverizzare', 'sbandare',
    'fluttuare', 'incrinare', 'sterminare', 'indirizzare', 'tratteggiare',
    'raffrontare', 'incartare', 'dirigere', 'caldeggiare', 'centralizzare',
    'attivare', 'sbucare', 'acquietare', 'intrecciare', 'uccidere',
    'svegliare', 'biasciare', 'sobillare', 'rabbonire', 'adempiere',
    'calamitare', 'esortare', 'pianificare', 'fingere', 'subordinare',
    'titillare', 'vuotare', 'costernare', 'governare', 'sfiancare',
    'irradiare', 'spaurire', 'spandere', 'finire', 'simulare', 'attrarre',
    'lavorare', 'dimenticare', 'coltivare', 'sbrigare', 'saccheggiare',
    'contornare', 'rinnegare', 'meditare', 'girovagare', 'balbettare',
    'sprecare', 'scialacquare', 'disturbare', 'assopire', 'placare',
    'assediare', 'emanare', 'contrariare', 'incitare', 'spronare',
    'annebbiare', 'sopire', 'falsificare', 'reclamizzare', 'avallare',
    'accontentare', 'girandolare', 'vistare', 'irritare', 'cullare',
    'sparpagliare', 'farfugliare', 'accendere', 'flagellare', 'prosciugare',
    'profondere', 'addolorare', 'impaurire', 'arrotolare', 'passeggiare',
    'appagare', 'spostare', 'trinciare', 'addolcire', 'attorniare', 'sferzare',
    'disertare', 'atterrare', 'trastullare', 'misconoscere', 'sgorgare',
    'atterrire', 'oppugnare', 'effondere', 'disaffezionare', 'riassettare',
    'annoiare', 'collazionare', 'inimicare', 'premere', 'riconfortare',
    'amministrare', 'borbottare', 'scornare', 'rassettare', 'scocciare',
    'divertire', 'lenire', 'affascinare', 'denigrare', 'staffilare',
    'estenuare', 'attizzare', 'decorare', 'trescare', 'stazzonare',
    'tranquillare', 'medicare', 'ciancicare', 'lordare', 'comparare',
    'incuriosire', 'immergere', 'straniare', 'documentare', 'ipnotizzare',
    'pavesare', 'intaccare', 'sconciare', 'richiamare', 'cambiare', 'umiliare',
    'inzuppare', 'sedurre', 'ammaliare', 'destare', 'attirare', 'raggiungere',
    'urtare', 'guarnire', 'convogliare', 'mandare', 'bisbigliare',
    'autenticare', 'programmare', 'assassinare', 'immalinconire', 'esalare',
    'istigare', 'abiurare', 'curare', 'tranciare', 'tracciare', 'vagolare',
    'raddolcire', 'sfinire', 'confrontare', 'indispettire','fare','avere','vivere'
]
```

```python
verbs = pd.DataFrame(verbs)
verbs.columns = ['verbs']
verbs.head(10)
```

```python
wc = wcbs.WordsClusterBySynonyms(verbs, 'verbs', lang='ita')
```

```python
df = wc.get_synonyms_pandas()
df.head(10)
```

```python
wc.plot_hist(df)
```
![img](https://ibb.co/ezJGNm)

```python
df = wc.set_treshold(20, df)
df.head(10)
```

```python
wc.plot_hist(df)
```

```python
matrix = wc.create_distance_matrix(criteria=min, verbose=True)
wc.plot_eps_ncluster(matrix, ntot=10, min_samples=6)
```

```python
result = wc.run_cluster(0.3,6, matrix)
```

```python
wc.plot_cluster_k(matrix, 'contraddire')
```
