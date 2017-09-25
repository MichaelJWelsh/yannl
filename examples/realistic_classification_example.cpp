/*
 * Realistic classification example.
 */
#include <yannl.hpp>
#include "print_matrix.hpp"

#include <iostream>
#include <string>
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <ctime>

using namespace std;
using namespace yannl;

void generate_artificial_training_set(Matrix &data, Matrix &target, size_t set_size);

int main() {
    /*
     * The goal of this network is to take in two numbers and determine if their
     * sum is positive or negative. Positive is represented by the matrix:
     * {1, 0}, while negative is represented by the matrix {0, 1}. The two
     * numbers should be in the domain: [-500, 500].
     */
    Matrix data, target;
    generate_artificial_training_set(data, target, 2000);

    // Set a unique seed for this training session.
    srand(time(NULL));

    /*
     * Create network with 2 input neurons, 2 hidden layers each with 8 neurons
     * and tanh activation function, and 2 output neurons with softmax
     * activation function.
     */
    Network network(CLASSIFICATION, 2, {8, 8}, {tanh, tanh}, 2, softmax);

    // Convenience structure for easy parameter filling.
    TrainingParams params;
    params.data = data;
    params.target = target;
    params.batch_size = data.rows();
    params.max_iterations = 1000;
    params.epoch_analysis_interval = 10;
    params.min_accuracy = 0.975;
    params.learning_rate = 0.1;
    params.annealing_factor = 50;
    params.regularization_factor = 0.0015;
    params.momentum_factor = 0.95;
    params.shuffle = true;

    // Train network. Request it to print out loss every 10 epochs.
    network.train(params, true);

    // Get network's predictions for new data and calculate accuracy.
    data = {
        {-134.5234, 500.123},
        {-234, -60.34},
        {52.23, 83},
        {320, -400},
        {200.1, 500},
        {-3, 4},
        {3, -4}
    };
    target = {
        {1, 0},
        {0, 1},
        {1, 0},
        {0, 1},
        {1, 0},
        {1, 0},
        {0, 1}
    };
    network.feed_forward(data);
    print_matrix(network.predict(true));
    printf("Accuracy: %.3f\n", network.accuracy(data, target));

    // Save and load it (for demonstration purposes).
    network.save("my_network.txt");
    network = Network("my_network.txt");

    // Give user opportunity to enter own data in command line for bleeding edge testing.
    string input;
    cout << endl << "Test network? (y\\n): ";
    cin >> input;
    cout << endl;
    transform(input.begin(), input.end(), input.begin(), ::tolower);
    if (input == "y") {
        cout << "Enter 2 numbers separated by a space and hit enter to get prediction of sum." << endl
                << "Numbers should be in the domain [-500, 500]; however, you can use numbers " << endl
                << "outside this range to test network's extrapolation capabilities." << endl
                << "Type in \"exit\" to exit." << endl << endl;
        while (true) {
            double first, second;
            cin >> input;
            transform(input.begin(), input.end(), input.begin(), ::tolower);
            if (input == "exit") {
                break;
            }
            first = stod(input);
            cin >> second;
            data = {{first, second}};
            network.feed_forward(data);
            if (network.predict(true) == Matrix{{1, 0}}) {
                cout << "POSITIVE" << endl << endl;
            } else {
                cout << "NEGATIVE" << endl << endl;
            }
        }
    }

    return 0;
}

void generate_artificial_training_set(Matrix &data, Matrix &target, size_t set_size) {
    // Generate data first.
    data = Matrix(set_size, 2);
    for (size_t i = 0; i < data.rows(); ++i) {
        double first, second;
        first = (rand() % 1000) - (rand() % 1000);
        second = (rand() % 1000) - (rand() % 1000);
        data[i][0] = first;
        data[i][1] = second;
    }

    // Generate target next.
    target = Matrix(set_size, 2);
    for (size_t i = 0; i < target.rows(); ++i) {
        double sum = 0;
        for (size_t j = 0; j < data.cols(); ++j) {
            sum += data[i][j];
        }

        if (sum > 0) {
            target[i][0] = 1;
        } else {
            target[i][1] = 1;
        }
    }
}
