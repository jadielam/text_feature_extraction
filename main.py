'''
Created on Jan 11, 2017

@author: dearj019
'''

from features import FeatureExtractor
from features import FeatureGeneralizer
from features import FeatureVectorizer

def main():
    fe = FeatureExtractor()
    fg = FeatureGeneralizer()
    fv = FeatureVectorizer()
    
    documents = ["I want to know you", "What is your name?", "GReat!", "I am drinking coffee in London, England"]
    list_of_docs_features = []
    for document in documents:
        features, tokens, entities = fe.extract_features(document, ["unigrams", "dependencies_l"])
        
        #other options
        #features, tokens, entities = fe.extract_features(document, ["bigrams"])
        #features, tokens, entities = fe.extract_features(document, ["dependencies_r"])
        #features, tokens, entities = fe.extract_features(document, ["dependencies_b"])
        generalized_features = fg.generalize_features(tokens, entities, features, ["stem", "lemma", "entity"])
        list_of_docs_features.append(generalized_features)
    
    vectorized_features = fv.vectorize_features(list_of_docs_features, "binary")
    #other options commented below
    #vectorized_features = fv.vectorize_features(list_of_docs_features, "tf")
    #vectorized_features = fv.vectorize_features(list_of_docs_features, "tf/length")
    print(list_of_docs_features)
    print(vectorized_features)

if __name__ == "__main__":
    main()