from neural_network import LossFunctions
from neural_network import datasets
from neural_network.classes.Optimizers import *
from neural_network.classes.Validation import *
from neural_network import utils
import pandas as pd
import wandb

wandb.init(project="neural_network")

# dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
# dataset_class_column = "class"
# number_inputs = len(dataset_attribute_columns)
loss_function = LossFunctions.MSE()
#
# dataset = datasets.read_monk1()[0]
# x_dataset = dataset[dataset_attribute_columns].to_numpy(dtype=np.float32)
# y_dataset = dataset[dataset_class_column].to_numpy()


dataset_attribute_columns = ["a1", "a2", "a3", "a4", "a5", "a6"]
dataset_class_column = "class"

df, dft = datasets.read_monk1()
df = pd.concat([df, dft], axis=0)
df = pd.get_dummies(df, columns=dataset_attribute_columns)
# dft = pd.get_dummies(dft, columns=dataset_attribute_columns)

y = df.pop("class").to_numpy()
X = df.to_numpy(dtype=np.float32)

perm = np.random.permutation(X.shape[0])
X = X[perm]
y = y[perm]

def model_builder(hp):
    layers = [
        InputLayer((None, X.shape[-1]), hp["units"], ActivationFunctions.Sigmoid(), initializer=Uniform(-0.1, 0.1)),
        # HiddenLayer(hp["units"], ActivationFunctions.Sigmoid(), initializer=Uniform(-0.1, 0.1)),
        OutputLayer(1, ActivationFunctions.Sigmoid(), initializer=Uniform(-0.1, 0.1))
    ]

    model = MLClassifier(
        layers=layers,
        loss=loss_function,
        optimizer=SGD(learning_rate=hp["learning_rate"], momentum=hp["momentum"], regularization=hp["regularization"]),
        batch_size=100,
        n_epochs=200,
        verbose=False
    )
    return model

hp = {"units": Hyperparameter(
    generator_logic="all_from_list",
    generator_space=[5, 10, 15],
    unfold=True),
    "learning_rate": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0.1],
        unfold=True),
    "momentum": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0.],
        unfold=True),
    "regularization": Hyperparameter(
        generator_logic="all_from_list",
        generator_space=[0., 0.0001, 0.01],
        unfold=True)
}
# tuner = TunerHO(ConfigurationGenerator(hp, mode="grid"), model_builder, validation_size=0.3, verbose=True)
tuner = TunerCV(ConfigurationGenerator(hp, mode="grid"), model_builder, n_fold=4, verbose=True)
tester = TesterCV(tuner, n_fold=4, verbose=True)

r = tester.fit(X, y)
r.dump("./dumps/test1.pickle")

r = TestResult.load("./dumps/test1.pickle")

r.validation_results[0].plot_one(0, "train_loss_curve", "val_loss_curve")
