
import itertools
import random
from neural_network.classes.Validation import Hyperparameter


class ConfigurationGenerator():
    """
    Given the rules for the hyperparmeters generation, you can iterate this objet to pass to the model builder a dict
    with a configuration, the rules are expressed are a dict{ hyperparemeter : possible_values } where possible 
    values are instances of Hyperparamenter class that generates an arbitrary number of values following some 
    creteia

    """
    def __init__(self, folded_hyper_space : dict, mode : str, num_trials : int = -1) -> dict:

        """
        :param folded_hyper_space: a dict where keys are strings and values are Hyperparameter objects i.e. they contains the rules to be iterated
        :param mode: grid or random, if grid the cartesian product will be used, if random a random value for each hyperparameter in each configuration generated
        :param num_trials: the number of configurations to generate, only used if mode is random search
        """

        # check of validity
        if mode=="grid":
            if num_trials != -1:
                raise Exception("In the grid search, the number of trials is the size of the cartesian product, so you cant indicate it")

            for key, value in folded_hyper_space.items():
                if not value.unfold:
                    raise Exception("Found an unfolded hyperparameter in a grid search, this will cause an infinite loop")
        
        self.configurations = []
        unfolded_hyper_space = {}
        
        if mode == "grid":
            
            for key, values in folded_hyper_space.items():
                unfolded_hyper_space[key] = [_ for _ in values]
      
            cartesian_iterator = itertools.product(*unfolded_hyper_space.values())
            configurations_list = [i for i in cartesian_iterator]
            self.configurations = [{
                a : b for (a, b) in zip(unfolded_hyper_space.keys(), configurations_list[j])  
                } 
                for j in range(len(configurations_list))
                ]
        
        if mode == "random":

            for key, values in folded_hyper_space.items():
                
                if values.unfold: # i can iterate 
                    unfolded_hyper_space[key] = [_ for _ in values]
                else: # i can't iterate (infinite)
                    unfolded_hyper_space[key] = values


            for i in range(num_trials):
                current_conf = {}
                for hp, values in unfolded_hyper_space.items():
                    if isinstance(values, list):
                        value = random.choice(values)
                        current_conf[hp] = value
                    elif not values.unfold:
                        value = next(values)
                        current_conf[hp] = value
                    else:
                        raise Exception("Unknown error")
                
                self.configurations.append(current_conf)         

    def __iter__(self):
        self.iter_index = -1
        self.iter_len = len(self.configurations)
        return self

    def __next__(self):
        self.iter_index += 1
        
        if self.iter_index == self.iter_len:
            raise StopIteration
        
        return self.configurations[self.iter_index]

