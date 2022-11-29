import ConfigurationGenerator

class Tuner(): # TODO still to complete

    def __init__(self, 
        train_set,
        val_set, 
        configurations : ConfigurationGenerator,
        model_builder
        ):

        self.train_set = train_set
        self.val_set = val_set
        self.configurations = configurations
        self.model_builder = model_builder
        self.results = []
    
    def run(self): #TODO add cross validation and parallelize 

        for config in self.configurations:
            model = self.model_builder(config)
            model.fit(self.traning_set, self.validation_set)
            evaluation = model.evalueate()
            self.results.append(evaluation)

    def best_model(self):
        return self.best_model  

    def all_history(self):
        return self.all_history