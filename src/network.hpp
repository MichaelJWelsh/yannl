/*
 * Representation of a neural network.
 */
#ifndef YANNL_NETWORK_HPP
#define YANNL_NETWORK_HPP

#include "prereqs.hpp"
#include "activation.hpp"
#include "connection.hpp"
#include "layer.hpp"
#include "matrix.hpp"

namespace yannl {

// A quality-of-life structure for convenient network training.
struct TrainingParams final {
    // Training data where each row is an entry with data.cols() features.
    Matrix data;

    // Expected output values with respect to given training data.
    Matrix target;

    // Size of each batch (1 for stochastic gradient descent, data.rows() for
    // batch gradient descent, or a multiple of data.rows() for mini-batch
    // gradient descent).
    std::size_t batch_size;

    // Max number of epochs to run the algorithm for.
    std::size_t max_iterations;

    // Interval that determines how many epochs are required before training
    // algorithm can write out verbose message (if verbose is true) and
    // calculate current accuracy (which is then compared with the minimum
    // requested accuracy to see if network can return early).
    std::size_t epoch_analysis_interval;

    // Minimum accuracy required before training algorithm can return early (between 0 and 1 inclusive).
    double min_accuracy;

    // Initial learning rate.
    double learning_rate;

    // Controls how slowly learning rate decreases (0 to not use annealing).
    double annealing_factor;

    // Regularization factor for L2 regularization.
    double regularization_factor;

    // Momentum factor (between 0 and 1 inclusive).
    double momentum_factor;

    // Shuffle or not to shuffle between epochs.
    bool shuffle;
};

// The network type.
enum NetworkType {
    CLASSIFICATION,
    REGRESSION
};

// Neural network representation.
class Network final {
public:
    // Creation.
    Network(NetworkType type,
            std::size_t num_features,
            const std::vector<std::size_t> &hidden_sizes,
            const std::vector<Activation> &hidden_activations,
            std::size_t num_outputs,
            Activation output_activation);
    explicit Network(const std::string &file_path);
    Network(const Network &rhs);
    Network(Network &&rhs);

    // Operator overloads.
    Network& operator=(const Network &rhs);
    Network& operator=(Network &&rhs);

    // Back-end accessors.
    NetworkType type() noexcept;

    // General purpose operations.
    void feed_forward(const Matrix &data);
    Matrix predict(bool one_hot_encoding = false);
    double accuracy(const Matrix &data, const Matrix &target);
    void save(const std::string &file_path);

    // Training.
    void train(const TrainingParams& params, bool verbose = false, std::ostream &verbose_output_stream = std::cout);
    void train(Matrix data,
            Matrix target,
            std::size_t batch_size,
            std::size_t max_iterations,
            std::size_t epoch_analysis_interval,
            double min_accuracy,
            double learning_rate,
            double annealing_factor,
            double regularization_factor,
            double momentum_factor,
            bool shuffle,
            bool verbose = false,
            std::ostream &verbose_output_stream = std::cout);

private:
    NetworkType _type;
    std::vector<internal::Layer> layers;
    std::vector<internal::Connection> connections;

    double loss_function(const Matrix &prediction, const Matrix &actual, double regularization_factor);
};

} // namespace yannl

#endif // YANNL_NETWORK_HPP
