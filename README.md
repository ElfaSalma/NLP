# 📌 Projet : Détection de Discours Haineux sur Twitter avec NLP

## 📖 Description
Ce projet vise à analyser et classifier des tweets pour détecter la présence de discours haineux à l'aide du **Traitement du Langage Naturel (NLP)**. Il comprend plusieurs étapes allant de l'exploration des données à la mise en place de modèles avancés de machine learning et deep learning.

## 🛠 Technologies utilisées
- **Python**
- **Bibliothèques** : `sklearn`, `pandas`, `numpy`, `matplotlib`, `nltk`, `spacy`, `gensim`, `TensorFlow`, `PyTorch`
- **Modèles de classification** : SVM, Naive Bayes, Réseaux de neurones (LSTM, BERT)
- **Vectorisation** : TF-IDF, Word Embeddings (Word2Vec, GloVe)

---

## 📊 Étapes du projet

### 🔹 1. Exploration des données
- Utilisation d'un dataset annoté pour la détection du discours haineux.
- Analyse de la distribution des classes (discours haineux vs non haineux).
- Étude des caractéristiques textuelles : longueur des tweets, mots fréquents, hashtags.

### 🔹 2. Prétraitement des tweets
- Nettoyage des tweets (suppression des emojis, mentions, liens, hashtags).
- Tokenisation et suppression des stopwords avec `nltk`.
- Lemmatisation avec `WordNetLemmatizer` et `spacy`.
- Représentation vectorielle avec **TF-IDF** et **Word Embeddings** (Word2Vec, GloVe).

### 🔹 3. Modélisation
- Implémentation de plusieurs modèles de classification :
  - **Machine Learning** : SVM, Logistic Regression, Naive Bayes.
  - **Deep Learning** : Réseaux de neurones (LSTM, Transformers type BERT).
- Comparaison des modèles à l’aide de métriques : accuracy, precision, recall, F1-score.

### 🔹 4. Analyse avancée et Text Mining
- Analyse du ton des tweets (sentiment analysis : positif, négatif, neutre).
- Implémentation de modèles avancés basés sur **Transformers (BERT)** pour améliorer la classification.

### 🔹 5. Évaluation des résultats
- Utilisation d'une matrice de confusion pour visualiser les résultats.
- Calcul et interprétation des courbes **ROC-AUC** pour évaluer la performance des modèles.


---


