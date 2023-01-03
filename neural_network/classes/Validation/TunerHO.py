import numpy as np

from neural_network.classes.Validation import ConfigurationGenerator
from neural_network.classes.Results import ValidationResult, ResultCollection
from concurrent.futures import ProcessPoolExecutor
from neural_network.utils import split_samples

class TunerHO:
    """
    Class to perform hold out validation for model selection
    """
    def __init__(self,
                 configurations: ConfigurationGenerator,
                 model_builder,
                 val_size: float,
                 verbose: bool,
                 default_metric: str = None,
                 default_reverse: bool = None,
                 shuffle: bool = True
                 ):
        """
        Args:
            configurations (ConfigurationGenerator): object implementing iterator interface to
                iterate over hyperparameter configuration
            model_builder (callable): a function that takes a dictionary that containing the hyperparameter
                configuration and return the model to validate
            val_size (float): dimension of validation set (in percentual)
            verbose (bool): print for every configuration of hyperparameters tried the value of the hyperparameters
        """
        self.configurations = configurations
        self.model_builder = model_builder
        self.val_size = val_size
        self.results = None
        self.verbose = verbose
        self.default_metric = default_metric
        self.default_reverse = default_reverse
        self.shuffle = shuffle

    @staticmethod
    def fit_configuration(params):
        config, model_builder, trainval_inputs, trainval_outputs, val_size, verbose, shuffle = params
        if verbose:
            print("Building model with the following configuration:", config)

        model = model_builder(config)
        if not model:
            return None

        if shuffle:
            perm = np.random.permutation(len(trainval_inputs))
            trainval_inputs = trainval_inputs[perm]
            trainval_outputs = trainval_outputs[perm]

        tr_inputs, vl_inputs, _ = split_samples(trainval_inputs, tr_size=1-val_size, vl_size=val_size, ts_size=0.)
        tr_outputs, vl_outputs, _ = split_samples(trainval_outputs, tr_size=1-val_size, vl_size=val_size, ts_size=0.)

        result = model.fit(
            tr_inputs,
            tr_outputs,
            [vl_inputs, vl_outputs]
        )
        return ValidationResult(config, [result])

    #self.results.add_result(fold_results)

    def fit(self, trainval_inputs, trainval_outputs):  # TODO  parallelize
        """
        Do the model selection using hold out validation to measure metrics on validation set
        Return:
            ResultCollection: the collection of all results obtained during the model selection
        """
        self.results = ResultCollection()

        configurations_params = [
            (
                config,
                self.model_builder,
                trainval_inputs,
                trainval_outputs,
                self.val_size,
                self.verbose,
                self.shuffle
            )
            for config in self.configurations
        ]
        with ProcessPoolExecutor() as executor:
            configurations_validation_result = executor.map(TunerHO.fit_configuration, configurations_params)

        # collecting the results for each runned configuration
        for configuration_results in configurations_validation_result:
            if configuration_results:
                self.results.add_result(configuration_results)
        return self.results


    def best_params(self, metric=None, reverse=None):
        """
        Get the best hyperparameters according to the specified metric
        """
        if not metric:
            metric = self.default_metric
        if not reverse:
            reverse = self.default_reverse
        self.results.sort(metric, reverse=reverse)
        a = self.results.list[0]
        a = a.hp_config
        return a

    def best_model(self, metric=None, reverse=None):
        """
        Get a (non-trained) new instance of the model with the best hyperparameters according to the specified metric
        """
        if metric is None and self.default_metric is None:
            raise ValueError("no metric provided to select best model")
        if reverse is None and self.default_reverse is None:
            raise ValueError("is not specified if the metric have to be maximised or minimized")
        return self.model_builder(self.best_params(metric, reverse))
