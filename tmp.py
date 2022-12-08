#####

# GENERIC SEARCH

####

def hyperparameters_generator(diz):
	"""
	resolvers the hp.choices etc. and yields the configurations
	"""
	for param, param_values in diz:
		if isinstance(param_values, hp.Choice):
			for value in param_values:
				yield {param: value}
	

def model_builder(hp):
	"""
	(provided by the user)
	builds the model using a configuration
	"""
	## param = "unit_1", param_value = 4
	if param == "unit_1":
		new_layer = HiddenLayer(n_units=param_value)
		
	layers = [new_layer]
	classifier = MLClassifier(layers)
	return classifier 

dict = {
	"unit_1": hp.Choice([4, 10, 16]), 
	"learning_rate": hp.Float([0.0001, 0.1], 5), 
	"unit_2": hp.Int([30, 50], 2)
}

def tuner(training_set, validation_set, dict, model_builder):
	results = []
	for hp in hyperparameters_generator(dict): #TODO
		model = model_builder(hp)
		model.fit(traning_set, validation_set)
		evaluation = model.evalueate()
		results.append(evaluation)
	
	
best : model, others : list(tuple) = tuner(dict, model_builder)
	
class Tuner:
	def __init__(self):
		pass
		
	def 
	
	
#######

## K-FOLD

#######

inputs=[0,1,2,3,4]
k_fold = K_Fold(len(inputs),n_split=3)

for (training_indexes, validation_indexes) in k_fold.get_folds():
	train_set = inputs[training_indexes]
	validation_set = inputs[validation_indexes]
	train_label = outputs[training_indexes]
	validation_labels = outputs[validation_indexes]


##

int/float(range, numbers_to_generate)  #prodice numbers_to_generate elements from a range

Choice([list]) #produce all elements

RandomChoice([list], numbers_to_generate) #prodice numbers_to_generate elements from a list