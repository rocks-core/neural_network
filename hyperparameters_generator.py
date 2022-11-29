import random
import itertools


class Hyperparameter():

    def __init__(
        self,
        generator_logic, # can be "random_choice_form_list","random_choice_form_range", "all_from_list"
        generator_space,
        random_elements_to_generate = -1
        ):
        
        # control of validity

        if(generator_logic != "random_choice_from_range"): 
            if (not isinstance(generator_space, list)) or (len(generator_space) == 0):
                print(f"ERROR: If generator_logic is <<{generator_logic}>>, generator_space should be a non empty list")
        
        if(generator_logic == "random_choice_from_range"):
            if (not isinstance(generator_space, tuple)) or (len(generator_space) != 2):
                print(f"ERROR: If generator_logic is <<{generator_logic}>>, generator_space should be a two element tuple")

        if(generator_logic != "all_from_list") and (random_elements_to_generate < 1):
            print(f"ERROR: If generator_logic is <<{generator_logic}>>, the number of random elements to generare should be > 0")

        
        # unfolding of the rules

        if generator_logic == "all_from_list":
            self.unfolded = generator_space
        
        if generator_logic == "random_choice_from_list":
            self.unfolded = random.sample(generator_space, random_elements_to_generate)

        if generator_logic == "random_choice_from_range":
            if isinstance(generator_space[0],float) or isinstance(generator_space[1],float):
                self.unfolded = [random.uniform(generator_space[0], generator_space[1]) for _ in range(random_elements_to_generate)]
            else:
                self.unfolded = random.sample(range(generator_space[0], generator_space[1]), random_elements_to_generate)


        self.iter_index = -1
        self.iter_len = len(self.unfolded)

        
    def __iter__(self):
        return self

    def __next__(self):
        self.iter_index += 1
        
        if self.iter_index == self.iter_len:
            raise StopIteration
        
        return self.unfolded[self.iter_index]


class ConfigurationGenerator():
    """
    Given the rules for the hyperparmeters generation, you can iterate this objet to pass to the model builder a dict
    with a configuration, the rules are expressed are a dict{ hyperparemeter : possible_values } where possible 
    values are instances of Hyperparamenter class that generates an arbitrary number of values following some 
    creteia, the possible keys for the hyper_space are 
    listed below

    """
    def __init__(self, folded_hyper_space : dict) -> dict:

        unfolded_hyper_space = {}

        for key, value in folded_hyper_space.items():
            #value is an istance of Hyperparamenter class, simply unfold with iteration
            unfolded_hyper_space[key] = [_ for _ in value]
        
        cartesian_iterator = itertools.product(*unfolded_hyper_space.values())
    
        configurations_list = [i for i in cartesian_iterator]\

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



## test
for i in ConfigurationGenerator(
    {
        "num_layers" : Hyperparameter(
            generator_logic = "all_from_list", 
            generator_space = [4, 5, 6, 7]),
        
        "lambda" : Hyperparameter(
            generator_logic = "random_choice_from_range", 
            generator_space = (0.1, 0.6), 
            random_elements_to_generate = 2),
    }
    ):
    
    print(i)