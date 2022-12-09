from neural_network import LossFunctions
from neural_network import datasets
from neural_network.classes.Optimizers import NesterovSGD
from neural_network.classes.Validation import *


dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
dataset_class_column = "class"
number_inputs = len(dataset_attribute_columns)
loss_function = LossFunctions.MSE()

dataset = datasets.read_monk1()[0]
x_dataset = dataset[dataset_attribute_columns].to_numpy(dtype=np.float32)
y_dataset = dataset[dataset_class_column].to_numpy()

def model_builder(hp):
    layers = [
        InputLayer((None, x_dataset.shape[-1]), x_dataset.shape[-1], ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        HiddenLayer(hp["units"], ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1)),
        OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-1, 1))
    ]

    model = MLClassifier(
        layers=layers,
        loss=loss_function,
        optimizer=NesterovSGD(learning_rate=hp["learning_rate"], momentum=hp["momentum"], regularization=hp["regularization"]),
        batch_size=100,
        n_epochs=100,
        verbose=False
    )
    return model

hp = {"units": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[6, 15]),
    "learning_rate": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0.1, 1.]),
    "momentum": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0.9]),
    "regularization": Hyperparameter(
        generator_logic="random_choice_from_range",
        generator_space=(0.0000001, 0.001),
        random_elements_to_generate=3)
}
tuner = TunerCV(ConfigurationGenerator(hp, mode="grid", num_trials=10), model_builder, n_fold=4, verbose=True)


tester = TesterCV(tuner, 4, True)
a = tester.fit(x_dataset, y_dataset)
a.dump("./dumps/test1.pickle")

a = TestResult.load("./dumps/test1.pickle")

a.validation_results[0].plot_one(0, "train_loss_curve", "val_loss_curve")