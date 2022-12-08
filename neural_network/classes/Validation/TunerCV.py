from neural_network.classes.Validation import ConfigurationGenerator
from neural_network.classes.Validation.K_fold import K_fold
from neural_network.classes.Results import FoldResult, ResultCollection


class TunerCV:  # TODO still to complete

    def __init__(self,
                 configurations: ConfigurationGenerator,
                 model_builder,
                 n_fold: int,
                 verbose: bool
                 ):
        self.configurations = configurations
        self.model_builder = model_builder
        self.n_fold = n_fold
        self.results = None
        self.verbose = verbose

    def fit(self, trainval_inputs, trainval_outputs):  # TODO  parallelize
        k_fold = K_fold(trainval_inputs.shape[0], self.n_fold)
        self.results = ResultCollection()
        for config in self.configurations:

            if self.verbose:
                print("Building model with the following configuration:", config)

            fold_results = []
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

            fold_results = FoldResult(config, fold_results)

            self.results.add_result(fold_results)
        return self.results


    def best_params(self, metric, reverse):
        self.results.sort(metric, reversed=reverse)
        a = self.results.list[0]
        a = a.hp_config
        return a

    def best_model(self, metric, reverse):
        return self.model_builder(self.best_params(metric, reverse))

    def all_history(self):
        return self.all_history
