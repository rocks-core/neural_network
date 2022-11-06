from neural_network.classes.Functions import ActivationFunctions
from neural_network import MLClassifier
from neural_network import datasets


if __name__ == "__main__":
    number_inputs = 2
    layer_sizes = (2, 2, 1)
    tr_df, vl_df, _ = datasets.circle()
    n_trials = 5

    tr_inputs = tr_df[["x", "y"]].to_numpy()
    tr_outputs = tr_df[["class"]].to_numpy()
    vl_inputs = vl_df[["x", "y"]].to_numpy()
    vl_outputs = vl_df[["class"]].to_numpy()

    trials = []
    for _ in range(n_trials):
        classifier = MLClassifier(
            number_inputs=2,
            layer_sizes=layer_sizes,
            activation_functions=(
                ActivationFunctions.Linear(),
                ActivationFunctions.Linear(),
                ActivationFunctions.Sigmoid()
            ),
            #regularization_term=0.01,
            batch_size=80,
            learning_rate=0.001,
            n_epochs=200,
            verbose=True
        )
        # training model
        classifier.fit(tr_inputs, tr_outputs)
        print("Done training")

        # validating result
        correct_predictions = 0

        for (input, expected_output) in zip(vl_inputs, vl_outputs):
            input = input.reshape(1, -1)  # inputs have to be row vector shape (1, n)
            real_output = classifier.predict(input)[0, 0]  # output is a (1,1) matrix now
            # in future the output will be a (m,n) matrix with n number of output units
            # and m number of input patterns given to the net
            if round(real_output) == expected_output:
                correct_predictions += 1

        trials.append(100 * (correct_predictions / len(vl_df)))

    print(f"min: {min(trials)}")
    print(f"max: {max(trials)}")
    avg = lambda l: sum(l)/len(l) if len(l) != 0 else 0
    print(f"avg: {avg(trials)}")

