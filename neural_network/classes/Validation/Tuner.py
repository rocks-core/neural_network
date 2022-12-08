from neural_network.classes.Validation import ConfigurationGenerator
from neural_network.classes.Validation import K_fold

class Tuner(): # TODO still to complete

    def __init__(self, 
        trainval_set_input,
        trainval_set_output,
        configurations : ConfigurationGenerator,
        model_builder, 
        k_fold : K_fold,
        verbose : bool
        ):

        self.trainval_set_input = trainval_set_input
        self.trainval_set_output = trainval_set_output
        self.configurations = configurations
        self.model_builder = model_builder
        self.k_fold = k_fold
        self.results = []
        self.verbose = verbose
    
    def fit(self): #TODO  parallelize 

        for config in self.configurations:
            
            if self.verbose:
                print("Building model with the following configuration:", config)
            
            model = self.model_builder(config)
            if model is None:
                pass
            
            fold_results = []
            for (fold_tr_indexes, fold_vl_indexes) in self.k_fold.get_folds():
                fold_tr_inputs, fold_tr_outputs = self.trainval_set_input[fold_tr_indexes], self.trainval_set_output[fold_tr_indexes]
                fold_vl_inputs, fold_vl_outputs = self.trainval_set_input[fold_vl_indexes], self.trainval_set_output[fold_vl_indexes]
            
                model.fit(fold_tr_inputs, fold_tr_outputs)
                evaluation = model.evaluate(fold_vl_inputs, fold_vl_outputs)
                
                fold_results.append(evaluation)

            fold_mean = fold_results.mean() # TODO it depends on the output of model.evaluate()
            
            self.results.append( (config, fold_results) )

    
    def best_model(self):
        #TODO: logic to select the best model based on self.results
        return self.best_model  

    def all_history(self):
        return self.all_history