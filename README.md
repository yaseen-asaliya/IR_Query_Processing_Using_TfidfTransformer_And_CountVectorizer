# IR_Query_Processing_Using_TfidfTransformer_And_CountVectorizer

This application is a part of `Information Retreval` course, it takes a bath for a given folderand a query, it use `TfidfTransformer` and `CounterVectroizer` to make query processing and return top 5 documents relevant to the query, sorted by relevance.

  * After collecting the courpus, then created `vectorizer` vectore used to converts the courpos to lowercase, removes English stop words, includes all rare words (at least one occurrence), and limits the vocabulary to the top 1000 most frequent words.
  ```
vectorizer = CountVectorizer(lowercase=True, stop_words="english", min_df=1, max_features=1000)
  ```


* Using `calculate_tf_idf_matrix` function I calculated `TF-IDF` by pass the `courpos` to the `vectorizer` vectore to make matrix include frequent of each term them pass that matrix to `tfidf_transformer` to calcute `TF-IDF` score.
```
def calculate_tf_idf_matrix(courpus, vectorizer, tfidf_transformer):    # courpus -> [[t1, t2, t3], [t1, t2, t3], ....]
    courpus_matrix = vectorizer.fit_transform(courpus)                  # (document_id, word_index)  frequent 
    tfidf_matrix = tfidf_transformer.fit_transform(courpus_matrix)      # (document_id, word_index)  TF-IDF score 
    return tfidf_matrix
```

* Then I calclated `Cosine Similarity` by passing a vectorized query and our `tfidf_transformer` matrix using `cosine_similarity` function
```
cosine_similarity_result = list(cosine_similarity(vectorized_query, tfidf_matrix)[0])
```
* Last thing I retreve top 5 relevent document by ranking them using `rank_the_docs` function by their relevance.
```
def rank_the_docs(cosine_similarity, files_names): 
    cosine_similarity_as_dictionary_sorted = convert_cosine_similarity_to_sorted_dictionary(cosine_similarity, files_names)
        
    top_five_documents_indices = {k: v for k, v in list(cosine_similarity_as_dictionary_sorted.items())[:5]}

    return top_five_documents_indices
```
* Here you can see a simple demo for using the application.

![demo](https://github.com/yaseen-asaliya/IR_Query_Processing_Using_TfidfTransformer_And_CountVectorizer/assets/59315877/3e459a5a-2f5f-43c1-be84-98acd2f237f2)
