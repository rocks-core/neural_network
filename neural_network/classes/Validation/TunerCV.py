from neural_network.classes.Validation import ConfigurationGenerator
from neural_network.classes.Validation.K_fold import K_fold
from neural_network.classes.Results import ValidationResult, ResultCollection
from concurrent.futures import ProcessPoolExecutor

class TunerCV:
    """
    Class to perform K-fold cross validation for model selection
    """
    def __init__(self,
                 configurations: ConfigurationGenerator,
                 model_builder,
                 n_fold: int,
                 verbose: bool
                 ):
        """
        Args:
            configurations (ConfigurationGenerator): object implementing iterator interface to
                iterate over hyperparameter configuration
            model_builder (callable): a function that takes a dictionary that containing the hyperparameter
                configuration and return the model to validate
            n_fold (int): number of fold
            verbose (bool): print for every configuration of hyperparameters tried the value of the hyperparameters
        """
        self.configurations = configurations
        self.model_builder = model_builder
        self.n_fold = n_fold
        self.results = None
        self.verbose = verbose

    @staticmethod
    def fit_configuration(params):
        config, model_builder, trainval_inputs, trainval_outputs, n_fold, verbose = params
        if verbose:
            print("Building model with the following configuration:", config)

        fold_results = []
        k_fold = K_fold(trainval_inputs.shape[0], n_fold)
        for (fold_tr_indexes, fold_vl_indexes) in k_fold.get_folds():
            model = model_builder(config)

            fold_tr_inputs, fold_tr_outputs = trainval_inputs[fold_tr_indexes], trainval_outputs[fold_tr_indexes]
            fold_vl_inputs, fold_vl_outputs = trainval_inputs[fold_vl_indexes], trainval_outputs[fold_vl_indexes]

            result = model.fit(
                fold_tr_inputs,
                fold_tr_outputs,
                [fold_vl_inputs, fold_vl_outputs]
            )

            fold_results.append(result)

        return ValidationResult(config, fold_results)
        #self.results.add_result(fold_results)

    def fit(self, trainval_inputs, trainval_outputs):  # TODO  parallelize
        """
        Do the model selection using K-fold cross validation to measure metrics on validation set

        Return:
            ResultCollection: the collection of all results obtained during the model selection
        """
        self.results = ResultCollection()
        """
        for config in configurations:
            if self.verbose:
                print("Building model with the following configuration:", config)

            fold_results = []
            k_fold = K_fold(trainval_inputs.shape[0], self.n_fold)
            for (fold_tr_indexes, fold_vl_indexes) in k_fold.get_folds():
                model = self.model_builder(config)

                fold_tr_inputs, fold_tr_outputs = trainval_inputs[fold_tr_indexes], trainval_outputs[fold_tr_indexes]
                fold_vl_inputs, fold_vl_outputs = trainval_inputs[fold_vl_indexes], trainval_outputs[fold_vl_indexes]

                result = model.fit(
                    fold_tr_inputs,
                    fold_tr_outputs,
                    [fold_vl_inputs, fold_vl_outputs]
                )

                fold_results.append(result)

            fold_results = ValidationResult(config, fold_results)
        """
        configurations_params = [
            (
                config,
                self.model_builder,
                trainval_inputs,
                trainval_outputs,
                self.n_fold,
                self.verbose
            )
            for config in self.configurations
        ]
        with ProcessPoolExecutor() as executor:
            configurations_validation_result = executor.map(TunerCV.fit_configuration, configurations_params)

        # collecting the results for each runned configuration
        for configuration_results in configurations_validation_result:
            self.results.add_result(configuration_results)
        return self.results


    def best_params(self, metric, reverse):
        """
        Get the best hyperparameters according to the specified metric
        """
        self.results.sort(metric, reverse=reverse)
        a = self.results.list[0]
        a = a.hp_config
        return a

    def best_model(self, metric, reverse):
        """
        Get a (non-trained) new instance of the model with the best hyperparameters according to the specified metric
        """
        return self.model_builder(self.best_params(metric, reverse))
