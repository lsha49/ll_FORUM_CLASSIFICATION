from cofet.CofetEntry import CofetEntry

config = dict()
config['testSize'] = 0.2
config['file'] = 'TestData.csv'
config['labelName'] = 'Label'
config['numOfGramFeatures'] = 0
config['useTfidf'] = False
config['useUniGram'] = False
config['useBiGram'] = False
config['clf'] = 'rf_clf'


# create a new cofet object
cofeter = CofetEntry(config)

# load data from config file
cofeter.load()

# adapt data to proper format by preprocessing
cofeter.adapt()

# compose features
cofeter.compose()

# train a model
cofeter.clf()

# evaluate
cofeter.elv()



