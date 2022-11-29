
import itertools
import Hyperparameter


class ConfigurationGenerator():
    """
    Given the rules for the hyperparmeters generation, you can iterate this objet to pass to the model builder a dict
    with a configuration, the rules are expressed are a dict{ hyperparemeter : possible_values } where possible 
    values are instances of Hyperparamenter class that generates an arbitrary number of values following some 
    creteia

    """
    def __init__(self, folded_hyper_space : dict) -> dict:

        """
        :param folded_hyper_space: a dict where keys are strings and values are Hyperparameter objects i.e. they contains the rules to be iterated
        """

        unfolded_hyper_space = {}

        for key, value in folded_hyper_space.items():
            #value is an istance of Hyperparamenter class, simply unfold with iteration
            unfolded_hyper_space[key] = [_ for _ in value]
        
        cartesian_iterator = itertools.product(*unfolded_hyper_space.values())
    
        configurations_list = [i for i in cartesian_iterator]

        self.configurations = [{
             a : b for (a, b) in zip(unfolded_hyper_space.keys(), configurations_list[j])  
             } 
             for j in range(len(configurations_list))
             ]

        self.iter_index = -1
        self.iter_len = len(self.configurations)

    def __iter__(self):
        return self

    def __next__(self):
        self.iter_index += 1
        
        if self.iter_index == self.iter_len:
            raise StopIteration
        
        return self.configurations[self.iter_index]

