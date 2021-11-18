from models.ml_classifiers import ml_clf

config = dict()
config['testSize'] = 0.2
config['numOfGramFeatures'] = 0
config['file'] = 'StanfordOrigin.csv'

classifier = ml_clf(config)
classifier.nb_clf()



