/*
 * Classification example.
 */
#include <yannl.hpp>
#include "print_matrix.hpp"

#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace yannl;

int main() {
    /*
     * The goal of this network is to take in two numbers and determine if their
     * sum is positive, zero, or negative. If positive, output is {1, 0, 0}. If
     * zero, output is {0, 1, 0}. If negative, output is {0, 0, 1}.
     */
    Matrix data = {
        {1, 0.23}, {2.1, 2.7}, {-1, 1}, {5.6, -10}, {-21.4, 22},
        {501.3, -600}, {1.678, 2}, {3.2, -5}, {0, 0}, {-83, -1.543}
    };
    Matrix target = {
        {1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1, 0, 0},
        {0, 0, 1}, {1, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 0, 1}
    };

    // Set a unique seed for this training session.
    srand(time(NULL));

    /*
     * Create network with 2 input neurons, 1 hidden layer with 4 neurons and
     * sigmoid activation function, and 3 output neurons with softmax activation
     * function.
     */
    Network network(CLASSIFICATION, 2, {4}, {sigmoid}, 3, softmax);

    // Convenience structure for easy parameter filling.
    TrainingParams params;
    params.data = data;
    params.target = target;
    params.batch_size = data.rows();
    params.max_iterations = 5000;
    params.epoch_analysis_interval = 100;
    params.min_accuracy = 0.95;
    params.learning_rate = 0.16;
    params.annealing_factor = 3000;
    params.regularization_factor = 0.002;
    params.momentum_factor = 0.9;
    params.shuffle = true;

    // Train network. Request it to print out loss every 100 epochs.
    network.train(params, true);

    // Get network's predictions for training data and calculate accuracy.
    network.feed_forward(data);
    print_matrix(network.predict(true));
    printf("Accuracy: %.3f\n", network.accuracy(data, target));

    // Save and load it.
    network.save("my_network.txt");
    network = Network("my_network.txt");

    return 0;
}
