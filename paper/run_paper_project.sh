#!bin/bash
python paper/create_title_data.py
python paper/feat2vec_imdb.py
python paper/word2vec_imdb.py
python paper/evaluate_vec_similarity.py
python paper/rank_byDeepFeature.py
