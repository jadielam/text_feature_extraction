'''
Created on Jan 10, 2017

@author: dearj019
'''
import spacy
from sklearn.feature_extraction import DictVectorizer
from spacy.en import English
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

class FeatureExtractor():
    def __init__(self):    
        self._parser = English()
    
    def extract_features(self, sentence, feature_names_s):
        sentence_u = unicode(sentence, "utf-8")
        to_return = {}
        
        parsed = self._parser(sentence_u)
        tokens = [a.text for a in parsed]
        
        entities = [None] * len(tokens)
        for entity in parsed.ents:
            start = entity.start
            end = entity.end 
            for i in range(start, end):
                if entities[i] is None:
                    entities[i] = []
                entities[i].append(entity.label_)
        for i in range(len(entities)):
            if entities[i] is not None:
                entities[i] = ".".join(entities[i])
            else:
                entities[i] = tokens[i]

        if 'unigrams' in feature_names_s:
            to_return['unigrams'] = [(a.text, a.i) for a in parsed] 
        
        if 'bigrams' in feature_names_s:
            bigrams = []
            for i in range(len(parsed) - 1):
                bigrams.append(((parsed[i].text, parsed[i].i), (parsed[i + 1].text, parsed[i + 1].i)))
            to_return['bigrams'] = bigrams
        
        if 'dependencies_l' in feature_names_s:
            dependencies_l = []
            for token in parsed:
                dependencies_l.append((token.dep_, (token.head.text, token.head.i), "*"))
            to_return['dependencies_l'] = dependencies_l
        
        if 'dependencies_r' in feature_names_s:
            dependencies_r = []
            for token in parsed:
                dependencies_r.append((token.dep_, "*", (token.text, token.i)))
            to_return['dependencies_r'] = dependencies_r
        
        if 'dependencies_b' in feature_names_s:
            dependencies_b = []
            for token in parsed:
                dependencies_b.append((token.dep_, (token.head.text, token.head.i), (token.text, token.i)))
            to_return['dependencies_b'] = dependencies_b
        
        return to_return, tokens, entities

class FeatureVectorizer():
    def __init__(self):
        self.v = DictVectorizer()
    
    def vectorize_features(self, dict_list, count_type):
        '''
        dict_list (dictionary):
            Format of the dictionary is: (string, (string, [tuples]))
        count_type (string):
           Count types to be supported are: binary, tf, tf/idf, tf/length
        
        '''
        #2.Create dictionaries of position and value for each feature
        features_list = []
        for doc in dict_list:
            
            feature_value_d = {}
            for outer_name, outer_doc in doc.iteritems():
                for inner_name, feature_list in outer_doc.iteritems():
                    
                    for feature in feature_list:
                        
                        if not feature in feature_value_d:
                            feature_value_d[feature] = 0
                            
                        if count_type == 'binary':
                            feature_value_d[feature] = 1
                        if count_type == 'tf' or count_type == 'tf/length':
                            feature_value_d[feature] += 1
                        
                    if count_type == 'tf/length':
                        if len(feature_list) > 0:
                            for feature in feature_value_d:
                                feature_value_d[feature] /= len(feature_list)
            
            features_list.append(feature_value_d)
        
        #3. Convert the dictionaries into sparse vectors
        X = self.v.fit_transform(features_list)
        return X
    
class FeatureGeneralizer():
    def __init__(self):
        self._stemmer = SnowballStemmer("english")
        self._lemmatizer = WordNetLemmatizer()
        
    
    def generalize_features(self, tokens, entities, features, levels_s):
    
        '''
        So far supports the following generalizations in levels_s:
        -stem
        -lemma
        -
        '''
        
        to_return = {}
        if 'no_generalization' in levels_s:
            to_return['no_generalization'] = dict(features)
        
        substitutions = []        
            
        if 'stem' in levels_s:
            stems = [self._stemmer.stem(word) for word in tokens]
            substitutions.append(('stem', stems))
            
        if 'lemma' in levels_s:
            lemmas = [self._lemmatizer.lemmatize(word) for word in tokens]
            substitutions.append(('lemma', lemmas))
        
        if 'entity' in levels_s:
            substitutions.append(('entities', entities))
            
        if 'wordnet' in levels_s:
            pass
        
        
        for feature_name, feature_values in features.iteritems():
            to_return[feature_name] = {}
            for substitution in substitutions:
                name = substitution[0]
                values = substitution[1]
                new_feature_values = list(feature_values)
                
                for i in range(len(new_feature_values)):
                    feature = feature_values[i]
                    
                    if feature_name == "unigrams":
                        new_feature = (values[feature[1]])
                    elif feature_name == "bigrams":
                        new_feature = (values[feature[0][1]], values[feature[1][1]])
                    elif feature_name == "dependencies_l":
                        new_feature = (feature[0], values[feature[1][1]], "*")
                    elif feature_name == "dependencies_r":
                        new_feature = (feature[0], "*", values[feature[2][1]])
                        
                    elif feature_name == "dependencies_b":
                        new_feature = (feature[0], values[feature[1][1]], values[feature[2][1]])
                    else:
                        new_feature = feature
                    
                    new_feature_values[i] = new_feature
                
                to_return[feature_name][name] = new_feature_values
                
            
        return to_return
        
    
        