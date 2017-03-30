/*
 * Regression example.
 */
#include <yannl.hpp>
#include "print_matrix.hpp"

#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace yannl;

int main() {
    // The goal of this network is to take in two numbers and determine their sum.
    Matrix data = {
        {1, 0.23}, {2.1, 2.7}, {-1, 1}, {5.6, -10}, {-21.4, 22},
        {501.3, -600}, {1.678, 2}, {3.2, -5}, {0, 0}, {-83, -1.543}
    };
    Matrix target = {
        {1.23}, {4.8}, {0}, {-4.4}, {0.6},
        {-98.7}, {3.678}, {-1.8}, {0}, {-84.543}
    };

    // Set a unique seed for this training session.
    srand(time(NULL));

    /*
     * Create network with 2 input neurons, 2 hidden layers each with 5 neurons
     * and sigmoid activation function, and 1 output neuron with linear
     * activation function.
     */
    Network network(REGRESSION, 2, {5, 5}, {sigmoid, sigmoid}, 1, linear);

    // Convenience structure for easy parameter filling.
    TrainingParams params;
    params.data = data;
    params.target = target;
    params.batch_size = data.rows();
    params.max_iterations = 10000;
    params.epoch_analysis_interval = 75;
    params.min_accuracy = 0.95;
    params.learning_rate = 0.04;
    params.annealing_factor = 8000;
    params.regularization_factor = 0.001;
    params.momentum_factor = 0.9;
    params.shuffle = true;

    // Train network. Request it to print out loss every 75 epochs.
    network.train(params, true);

    // Get network's predictions for training data and calculate accuracy.
    network.feed_forward(data);
    print_matrix(network.predict());
    printf("Accuracy: %.3f\n", network.accuracy(data, target));

    // Save and load it.
    network.save("my_network.txt");
    network = Network("my_network.txt");

    return 0;
}
