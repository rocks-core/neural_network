from neural_network.classes.Validation import ConfigurationGenerator
from neural_network.classes.Validation import K_fold

class TunerCV: # TODO still to complete

    def __init__(self,
        configurations : ConfigurationGenerator,
        model_builder, 
        k_fold : K_fold,
        verbose : bool
        ):
        self.configurations = configurations
        self.model_builder = model_builder
        self.k_fold = k_fold
        self.results = []
        self.verbose = verbose
    
    def fit(self, trainval_inputs, trainval_outputs): #TODO  parallelize
        for config in self.configurations:
            
            if self.verbose:
                print("Building model with the following configuration:", config)

            fold_results = []
            for (fold_tr_indexes, fold_vl_indexes) in self.k_fold.get_folds():
                model = self.model_builder(config)

                fold_tr_inputs, fold_tr_outputs = trainval_inputs[fold_tr_indexes], trainval_outputs[fold_tr_indexes]
                fold_vl_inputs, fold_vl_outputs = trainval_inputs[fold_vl_indexes], trainval_outputs[fold_vl_indexes]
            
                result = model.fit(
                    fold_tr_inputs,
                    fold_tr_outputs,
                    [fold_vl_inputs, fold_vl_outputs]
                )
                
                fold_results.append(result)

            fold_mean = fold_results.mean() # TODO it depends on the output of model.evaluate()
            
            self.results.append( (config, fold_results) )

    
    def best_model(self):
        #TODO: logic to select the best model based on self.results
        return self.best_model  

    def all_history(self):
        return self.all_history