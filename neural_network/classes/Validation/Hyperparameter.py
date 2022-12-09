import random

class Hyperparameter():

    """
    This class allows you to generate an itarable based on rules that you decide

    If the unfold param is true, a list of values will be generated *at the creation* of this object, otherwise a 
    value based on the rules will be generated on demand. Of course if you want to unfold, you have to indicate the 
    number of elements of the new list.

    If the generation logic is "all_from_list", unfold will be forced to True

    """

    def __init__(
        self,
        generator_logic, # can be "random_choice_form_list","random_choice_form_range", "all_from_list"
        generator_space,
        unfold : bool = False, 
        elements_to_generate : int = -1
        ):

        """
        :param generator_logic: string that could be: {random_choice_from_range, random_choice_from_list, all_from_list}
        :param generator_space: a tuple or a list, based on the generator logic, tuple (with two values) are used for indicate a range 
        :param unfold: boolean, if false it will generate infinite values, if true the "elements_to_generate" should be indicated
        :param elements_to_genetate: number of elements to yeald when unfolding the hyperparameter
        """
        
        self.unfold = unfold
        
        # control of validity

        if generator_logic == "all_from_list" and not unfold:
            raise Exception("If you use 'all_from_lits' you have to set unfold=True")

        if unfold and generator_logic !="all_from_list" and  elements_to_generate == -1:
            raise Exception("If you want to unfold a random choice hyperparameter, you have to indicate the number of elements to generate")

        if not unfold and elements_to_generate != -1:
            raise Exception("If you don't unfold the hyperparameter, it doesn't make sense to indicate the 'element_to_generate' because it will depend of the grid serach settings")

    
        
        # unfolding of the rules

        if generator_logic == "all_from_list":
            self.unfolded = generator_space
        
        if self.unfold:
            if generator_logic == "random_choice_from_list":
                self.unfolded = random.sample(generator_space, elements_to_generate)

            if generator_logic == "random_choice_from_range":
                if isinstance(generator_space[0],float) or isinstance(generator_space[1],float):
                    self.unfolded = [random.uniform(generator_space[0], generator_space[1]) for _ in range(elements_to_generate)]
                else:
                    self.unfolded = random.sample(range(generator_space[0], generator_space[1]), elements_to_generate)
        
        elif not self.unfold:
            # save the parameters in the class in order to generate random values on demand
            self.generator_logic = generator_logic
            self.generator_space = generator_space


    def __iter__(self):

        #list already unfolded before the iteration
        if self.unfold:
            self.iter_index = -1
            self.iter_len = len(self.unfolded)
        
        return self

    def __next__(self):
        
        #list already unfolded before the iteration
        if self.unfold:
            self.iter_index += 1 
            if self.iter_index == self.iter_len:
                raise StopIteration 
            return self.unfolded[self.iter_index]
        
        elif not self.unfold:
            if self.generator_logic == "random_choice_from_list":
                return random.sample(self.generator_space, 1)
        
            if self.generator_logic == "random_choice_from_range":
                if isinstance(self.generator_space[0], float) or isinstance(self.generator_space[1], float):
                    return random.uniform(self.generator_space[0], self.generator_space[1])
                else:
                    return random.sample(range(self.generator_space[0], self.generator_space[1]), 1)
