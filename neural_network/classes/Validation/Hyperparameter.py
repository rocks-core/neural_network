import random

class Hyperparameter():

    """
    This class allows you to generate an itarable based on rules that you decide, the elements are generated without repetitions

    """

    def __init__(
        self,
        generator_logic, # can be "random_choice_form_list","random_choice_form_range", "all_from_list"
        generator_space,
        random_elements_to_generate = -1
        ):

        """
        :param generator_logic: string that could be: {random_choice_from_range, random_choice_from_list, all_from_list}
        :param generator_space: a tuple or a list, based on the generator logic, tuple (with two values) are used for indicate a range 
        :param random_element_to_generate: number or random element to generate from a range o to pick from a list 
        """
        
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
