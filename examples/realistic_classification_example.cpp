/*
 * Realistic classification example.
 */
#include <yannl.hpp>
#include "print_matrix.hpp"

#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace yannl;

void generate_artificial_training_set(Matrix &data, Matrix &target, size_t set_size);

int main() {
    /*
     * The goal of this network is to take in two numbers and determine if their
     * sum is positive, zero, or negative. If positive, output is {1, 0, 0}. If
     * zero, output is {0, 1, 0}. If negative, output is {0, 0, 1}.
     */
    Matrix data, target;
    generate_artificial_training_set(data, target, 2000);

    // Set a unique seed for this training session.
    srand(time(NULL));

    /*
     * Create network with 2 input neurons, 3 hidden layers each with 6 neurons
     * and sigmoid activation function, and 3 output neurons with softmax
     * activation function.
     */
    Network network(CLASSIFICATION, 2, {6, 6, 6}, {sigmoid, sigmoid, sigmoid}, 3, softmax);

    // Convenience structure for easy parameter filling.
    TrainingParams params;
    params.data = data;
    params.target = target;
    params.batch_size = data.rows();
    params.max_iterations = 1000;
    params.min_accuracy = 0.95;
    params.learning_rate = 0.16;
    params.annealing_factor = 3000;
    params.regularization_factor = 0.002;
    params.momentum_factor = 0.9;
    params.shuffle = true;

    // Train network. Request it to print out loss every 100 epochs.
    network.train(params, true);

    // Get network's predictions for new data and calculate accuracy.
    data = {
        {-134.5234, 500.123}, {-234, -60.34}, {-4.5, 4.5}
    };
    target = {
        {1, 0, 0}, {0, 0, 1}, {0, 1, 0}
    };
    network.feed_forward(data);
    print_matrix(network.predict(true));
    printf("Accuracy: %.3f\n", network.accuracy(data, target));

    // Save and load it.
    network.save("my_network.txt");
    network = Network("my_network.txt");

    return 0;
}

void generate_artificial_training_set(Matrix &data, Matrix &target, size_t set_size) {
    // Generate data first.
    data = Matrix(set_size, 2);
    for (size_t i = 0; i < data.rows(); ++i) {
        double first, second;
        if (i % 3 == 0) {
            first = rand() % 1000;
            second = -first;
        } else {
            first = (rand() % 1000) - (rand() % 1000);
            second = (rand() % 1000) - (rand() % 1000);
        }

        data[i][0] = first;
        data[i][1] = second;
    }

    // Generate target next.
    target = Matrix(set_size, 3);
    for (size_t i = 0; i < target.rows(); ++i) {
        double sum = 0;
        for (size_t j = 0; j < data.cols(); ++j) {
            sum += data[i][j];
        }

        if (sum > 0) {
            target[i][0] = 1;
        } else if (sum == 0) {
            target[i][1] = 1;
        } else if (sum < 0) {
            target[i][2] = 1;
        }
    }
}
