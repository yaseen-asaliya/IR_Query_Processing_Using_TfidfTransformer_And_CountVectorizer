import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def collecting_courpus(folder_path): 
    documents_data = []
    files_names = os.listdir(folder_path)
    for file in files_names:
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file),'r') as file:
                text = file.read()
                documents_data.append(text)
    return documents_data, files_names

def calculate_tf_idf_matrix(courpus, vectorizer, tfidf_transformer):    # courpus -> [[t1, t2, t3], [t1, t2, t3], ....]
    courpus_matrix = vectorizer.fit_transform(courpus)                  # (document_id, word_index)  frequent 
    tfidf_matrix = tfidf_transformer.fit_transform(courpus_matrix)      # (document_id, word_index)  TF-IDF score 
    return tfidf_matrix

def vectorize_the_query(query, tfidf_transformer, vectorizer):
    vectorized_query = tfidf_transformer.transform(vectorizer.transform([query]))
    return vectorized_query

def convert_cosine_similarity_to_sorted_dictionary(cosine_similarity, files_names):
    counter = 0
    cosine_similarity_as_dictionary = {}
    for val in cosine_similarity:
        cosine_similarity_as_dictionary.setdefault(counter+1, [files_names[counter],val]) # {ID: [file_name, cosine_similarity_result]}
        counter+=1

    # Sort dictionary by cosine similarity
    cosine_similarity_as_dictionary_sorted = {k: v for k, v in sorted(cosine_similarity_as_dictionary.items(), key=lambda item: item[1][1], reverse=True)}
    return cosine_similarity_as_dictionary_sorted

def rank_the_docs(cosine_similarity, files_names): 
    cosine_similarity_as_dictionary_sorted = convert_cosine_similarity_to_sorted_dictionary(cosine_similarity, files_names)
        
    top_five_documents_indices = {k: v for k, v in list(cosine_similarity_as_dictionary_sorted.items())[:5]}

    return top_five_documents_indices

def print_top_5_related_document(ranked_documents, folder_path):
    print("\nHere are the top 5 documents relevant to your query, sorted by relevance from highest to lowest:")
    print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    for doc in ranked_documents.values():
        print(f"Document Name: {doc[0]}")
        print(f"Path: {folder_path}\{doc[0]}")
        print(f"Cosine Similarity: {doc[1]}")
        print("--------------------------------------------\n")

        
    
def demo():
    folder_path = input("\nEnter the path of your folder: ")
    query =  input("Enter the your query: ")
    
    # This vectore converts the courpos to lowercase, removes English stop words, includes all rare words (at least one occurrence), and limits the vocabulary to the top 1000 most frequent words.
    vectorizer = CountVectorizer(lowercase=True, stop_words="english", min_df=1, max_features=1000)
    tfidf_transformer = TfidfTransformer()

    courpus, files_names  = collecting_courpus(folder_path) 
    
    tfidf_matrix = calculate_tf_idf_matrix(courpus, vectorizer, tfidf_transformer)
    vectorized_query = vectorize_the_query(query, tfidf_transformer, vectorizer)
    
    cosine_similarity_result = list(cosine_similarity(vectorized_query, tfidf_matrix)[0])
    ranked_documents = rank_the_docs(cosine_similarity_result, files_names)

    print_top_5_related_document(ranked_documents, folder_path)
    
        
        
if __name__ == "__main__":
    demo()




