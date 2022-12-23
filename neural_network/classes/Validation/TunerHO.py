# from neural_network.classes.Validation import ConfigurationGenerator
# from neural_network.classes.Results import ValidationResult, ResultCollection
#
# from sklearn.model_selection import train_test_split
#
# class TunerHO:
#     """
#     Class to perform hold out validation for model selection
#     """
#     def __init__(self,
#                  configurations: ConfigurationGenerator,
#                  model_builder,
#                  validation_size,
#                  verbose: bool
#                  ):
#         """
#         Args:
#             configurations (ConfigurationGenerator): object implementing iterator interface to
#                 iterate over hyperparameter configuration
#             model_builder (callable): a function that takes a dictionary that containing the hyperparameter
#                 configuration and return the model to validate
#             validation_size (float): ratio between validation set and design set
#             verbose (bool): print for every configuration of hyperparameters tried the value of the hyperparameters
#         """
#         self.configurations = configurations
#         self.model_builder = model_builder
#         self.validation_size = validation_size
#         self.results = None
#         self.verbose = verbose
#
#     def fit(self, trainval_inputs, trainval_outputs):  # TODO  parallelize
#         """
#         Do the model selection using Hold out validation to measure metrics on validation set
#
#         Return:
#             ResultCollection: the collection of all results obtained during the model selection
#         """
#         self.results = ResultCollection()
#         for config in self.configurations:
#
#             if self.verbose:
#                 print("Building model with the following configuration:", config)
#
#
#             model = self.model_builder(config)
#
#             fold_tr_inputs, fold_vl_inputs, fold_tr_outputs, fold_vl_outputs = train_test_split(trainval_inputs, trainval_outputs, test_size=self.validation_size)
#
#             result = model.fit(
#                 fold_tr_inputs,
#                 fold_tr_outputs,
#                 [fold_vl_inputs, fold_vl_outputs]
#             )
#
#             fold_results = ValidationResult(config, [result])
#
#             self.results.add_result(fold_results)
#         return self.results
#
#
#     def best_params(self, metric, reverse):
#         """
#         Get the best hyperparameters according to the specified metric
#         """
#         self.results.sort(metric, reverse=reverse)
#         a = self.results.list[0]
#         a = a.hp_config
#         return a
#
#     def best_model(self, metric, reverse):
#         """
#         Get a (non-trained) new instance of the model with the best hyperparameters according to the specified metric
#         """
#         return self.model_builder(self.best_params(metric, reverse))
#
