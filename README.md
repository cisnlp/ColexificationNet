# ColexificationNet

We use [Conceptualizer](https://github.com/yihongL1U/conceptualizer) to extract the colexification patterns directly from a parallel corpora. This repositories contain a more effiecient version of **Conceptualizer** (forward pass + backward pass) introduced in [Conceptualizer paper](https://arxiv.org/abs/2305.08475).


```
.
├── README.md
├── association_finder.py
├── eva
│   ├── baseline_vectors
│   │   ├── other_word_vectors_process.ipynb
│   │   ├── sentence_classification.ipynb
│   │   ├── sentence_retrieval.ipynb
│   │   └── train_sentence_ID_script.py
│   ├── colexification_patterns
│   │   ├── clics_neighbors_dict.pickle
│   │   ├── conceptualizer_100_neighbors_dict.pickle
│   │   ├── conceptualizer_10_neighbors_dict.pickle
│   │   ├── conceptualizer_1_neighbors_dict.pickle
│   │   ├── conceptualizer_20_neighbors_dict.pickle
│   │   ├── conceptualizer_50_neighbors_dict.pickle
│   │   ├── conceptualizer_5_neighbors_dict.pickle
│   │   └── eva_colexification.ipynb
│   ├── round_trip
│   │   ├── round_trip.py
│   │   └── round_trip_min_langs.py
│   ├── sentence_classification
│   │   ├── sentence_classification.ipynb
│   │   └── sentence_classification.py
│   └── sentence_retrieval
│       ├── sentence_retrieval.ipynb
│       ├── sentence_retrieval.py
│       └── test_ids.txt
├── network_related
│   ├── __init__.py
│   ├── eflomal_network_builder.py
│   ├── eflomal_training.py
│   ├── iso2area.pickle
│   ├── iso2family.pickle
│   ├── network_builder.py
│   ├── train_different_min_language_embedding.py
│   └── updated
│       ├── NetworkAnalysis-Basic.ipynb
│       └── NetworkAnalysis-LanguageFamilies-Areas.ipynb
├── processing_concepts.py
├── processing_concepts_eng.py
├── processing_ngrams.py
└── processing_parallel.py
```


## Colexification extraction pipeline

(1) Preprocess the parallel data to obtain all ngrams for each verse in all languages:
```
python -u processing_ngrams.py --updated_ngrams true --ignore_case true 
```

(2) Create parallel data that are required in the subsequent computation:
```
python -u processing_parallel.py --updated_ngrams true --src_lang eng
```

(3) Obtain valid concepts and their statistics (for English,  we use lemmata as concepts):
```
python -u processing_concepts.py --updated_ngrams true --src_lang eng --ignore_case true
```

(4) run the following command to extract colexifications by Conceptualizer:
```
python -u association_finder.py --updated_ngrams true --do_lemmatize true --lemmatizer_type spacy --ignore_case true --src_lang eng --use_multiprocessing true
```

## Building networks

### Prelimilary step
For the first time, go to the directory and run python code like the following:
```
cd network_related
```

```
from network_builder import ConceptNetwork
considered_lang = 'all'
net = ConceptNetwork(involved_lang=considered_lang, load_directed_graph_from_path=False, use_updated=True)
net.store_net()
```

The codes above will store a **directed** network based on the colexification patterns of all languages.
Use the following code to directly load the network from the disk (if you have stored it using codes above):
```
net = ConceptNetwork(involved_lang=considered_lang, load_directed_graph_from_path=True, use_updated=True)
```

### Building ColexNet and ColexNet+

Specifying the minimum number of languages (e.g., 50) for an colexification edge to be included in both networks (both ColexNet and ColexNet+ are undirected networks) and run following code:
```
colexnet = net.to_undirected(aggregate_type='union', minimum_number_of_langs=50)
colexnet_plus = net.expand_graph(minimum_number_of_langs=50)
```

### Visualizations

To visualize the communities in ColexNet, please refer to `./network_related/updated/NetworkAnalysis-Basic.ipynb`.
  
Alternatively, please visit our online [demo](https://conceptexplorer.cis.lmu.de/) for visualizations of communities and the concepts in ColexNet.


### Embedding training

Simply run the following command to train multilingual embeddings on ColexNet+ for different hyperparameters (1, 5, 10, 20, 50 and 100 as number of minimum languages to preserve an colexification edge)
```
cd network_related
python -u train_different_min_language_embedding.py
```

You could find our published embeddings and networks [here](https://doi.org/10.5281/zenodo.7920596
).

## Evaluations

### Baselines

We use four strong baselines in our work: **sentence_id**, **clique_word**, **nt_word** and **eflomal-aligned**. 

The embeddings of **clique_word** and **nt_word** can be downloaded [here](http://cistern.cis.lmu.de/comult/).

Run the following code to train sentence_id embeddings:
```
cd eva
cd baseline_vectors
python -u train_sentence_ID_script.py
```

To generate eflomal-aligned embeddings

(1) first run the following codes to create an alignment graph and store it on the disk:
```
from network_builder import EflomalAlignmentNetwork
net = EflomalAlignmentNetwork(load_graph_from_path=False)
net.store_net()
net.store_vocab()
```

(2) then run the following command to train node embeddings:
```
cd network_related
python -u eflomal_training.py
```

### Colexification pattern identification

The ground-truth colexification patterns from [CLICS](https://clics.clld.org/) can be found at `./eva/colexification_patterns/clics_neighbors_dict.pickle` and coleixication patterns identified in ColexNet can found in `./eva/colexification_patterns/conceptualizer_50_neighbors_dict.pickle` (hyperparameter of 50, for example).  

Please then refer to `./eva/colexification_patterns/eva_colexification.ipynb` for codes for evaluation.


### Round-trip translations

The codes for round-trip translation can be found in `./eva/round_trip`.

Run the following for reproducing the results for round-trip translation of **sentence_id**, **clique_word**, **nt_word**, **eflomal-aligned** and **ColexNet2Vec** (50, by default) embeddings:
```
python -u round_trip.py
```

Run the following for reproducing the results for round-trip translation of **ColexNet2Vec** embeddings for different hyperparameters (1, 5, 10, 20, 50 and 100 as number of minimum languages to preserve an colexification edge) :
```
python -u round_trip_min_langs.py
```

### Sentence retrieval

The codes for sentence retrieval can be found in `./eva/sentence_retrieval`.

Run the following for processing the sentence retrieval dataset :
```
python -u sentence_retrieval.py
```

Please then refer to `./eva/sentence_retrieval/sentence_retrieval.ipynb` for codes of evaluation on ColexNe2Vec.  

Please then refer to `./eva/baseline_vectors/sentence_retrieval.ipynb` for codes of evaluation on other embeddings.  


### Sentence classification

The codes for sentence classification can be found in `./eva/sentence_classification`.

Run the following for processing the sentence classification dataset :
```
python -u sentence_classification.py
```

Please then refer to `./eva/sentence_retrieval/sentence_classification.ipynb` for codes of evaluation on ColexNe2Vec.  

Please then refer to `./eva/baseline_vectors/sentence_classification.ipynb` for codes of evaluation on other embeddings.  


## References

Please cite [[1]](https://arxiv.org/abs/2305.12818) and [[2]](https://arxiv.org/abs/2305.08475) if you found the resources in this repository useful.

### Crosslingual Transfer Learning for Low-Resource Languages Based on Multilingual Colexification Graphs
[1] Y. Liu, H. Ye, L. Weissweiler, H. Schuetze [*Crosslingual Transfer Learning for Low-Resource Languages Based on Multilingual Colexification Graphs*](https://arxiv.org/abs/2305.12818)
```
@article{liu2023colexnet,
  title={Transfer Learning for Low-Resource Languages Based on Multilingual Colexification Graphs},
  author={Liu, Yihong and Ye, Haotian and Weissweiler, Leonie and Schuetze, Hinrich},
  journal={arxiv},
  year={2023},
  url={https://arxiv.org/abs/2305.12818},
}
```

### A Crosslingual Investigation of Conceptualization in 1335 Languages
[2] Y. Liu, H. Ye, L. Weissweiler, P. Wicke, R. Pei, R. Zangenfeind, H. Schuetze [*A Crosslingual Investigation of Conceptualization in 1335 Languages*](https://aclanthology.org/2023.acl-long.726)

```
@inproceedings{liu-etal-2023-crosslingual,
    title = "A Crosslingual Investigation of Conceptualization in 1335 Languages",
    author = {Liu, Yihong  and
      Ye, Haotian  and
      Weissweiler, Leonie  and
      Wicke, Philipp  and
      Pei, Renhao  and
      Zangenfeind, Robert  and
      Sch{\"u}tze, Hinrich},
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.726",
    doi = "10.18653/v1/2023.acl-long.726",
    pages = "12969--13000",
}
```

