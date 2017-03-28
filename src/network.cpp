/*
 * Representation of a neural network.
 */
#include "network.hpp"

using namespace yannl::internal;
using std::size_t;
using std::string;
using std::ifstream;
using std::ofstream;
using std::ostream;
using std::numeric_limits;
using std::invalid_argument;
using std::round;
using std::log;
using std::pow;

namespace yannl {

// Creates network.
Network::Network(NetworkType type,
        size_t num_features,
        const std::vector<size_t> &hidden_sizes,
        const std::vector<Activation> &hidden_activations,
        size_t num_outputs,
        Activation output_activation) :
        _type(type)
{
    if (num_features == 0) {
        throw invalid_argument("Input layer must have at least one neuron!");
    } else if (num_outputs == 0) {
        throw invalid_argument("Output layer must have at least one neuron!");
    } else if (output_activation == nullptr) {
        throw invalid_argument("Output layer must have a callable activation function!");
    } else if (hidden_sizes.size() != hidden_activations.size()) {
        throw invalid_argument("Unable to deduce number of hidden layers!");
    }

    size_t num_layers = 2 + hidden_sizes.size();
    this->layers.reserve(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        if (i == 0) {
            this->layers.push_back(Layer(INPUT, num_features, nullptr));
        } else if (i == num_layers - 1) {
            this->layers.push_back(Layer(OUTPUT, num_outputs, output_activation));
        } else {
            this->layers.push_back(Layer(HIDDEN, hidden_sizes[i - 1], hidden_activations[i - 1]));
        }
    }

    size_t num_connections = num_layers - 1;
    this->connections.reserve(num_connections);
    for (size_t i = 0; i < num_connections; ++i) {
        this->connections.push_back(Connection(&this->layers[i], &this->layers[i + 1]));
    }
}

// Creates network from file.
Network::Network(const string &file_path) {
    ifstream input;
    input.open(file_path);
    if (!input) {
        throw invalid_argument("Unable to load network!");
    }

    // Get network type.
    NetworkType type;
    string stringified_network_type;
    input >> stringified_network_type;
    if (stringified_network_type == "CLASSIFICATION") {
        type = CLASSIFICATION;
    } else {
        type = REGRESSION;
    }

    // Get number of layers.
    size_t num_layers;
    input >> num_layers;

    // Get layer sizes.
    std::vector<size_t> layer_sizes(num_layers);
    for (size_t i = 0; i < num_layers; ++i) {
        input >> layer_sizes[i];
    }

    // Get activation functions.
    std::vector<Activation> activation_funcs(num_layers - 1);
    for (size_t i = 0; i < num_layers - 1; ++i) {
        string temp;
        input >> temp;
        activation_funcs[i] = get_activation_by_name(temp);
    }

    // Network creation.
    size_t num_features = layer_sizes[0];
    size_t num_outputs = layer_sizes[num_layers - 1];
    size_t num_hidden_layers = num_layers - 2;
    Activation output_activation = activation_funcs[num_hidden_layers];
    if (num_hidden_layers > 0) {
        std::vector<size_t> hidden_sizes(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            hidden_sizes[i] = layer_sizes[i + 1];
        }
        std::vector<Activation> hidden_activations(num_hidden_layers);
        for (size_t i = 0; i < num_hidden_layers; ++i) {
            hidden_activations[i] = activation_funcs[i];
        }
        *this = Network(type, num_features, hidden_sizes, hidden_activations, num_outputs, output_activation);
    } else {
        *this = Network(type, num_features, {}, {}, num_outputs, output_activation);
    }

    // Get weights.
    for (size_t k = 0; k < this->connections.size(); ++k) {
        Connection &connection = this->connections[k];
        for (size_t i = 0; i < connection.weights.rows(); ++i) {
            for (size_t j = 0; j < connection.weights.cols(); ++j) {
                input >> connection.weights[i][j];
            }
        }
    }

    // Get bias.
    for (size_t k = 0; k < this->connections.size(); ++k) {
        Connection &connection = this->connections[k];
        for (size_t j = 0; j < connection.bias.cols(); ++j) {
            input >> connection.bias[0][j];
        }
    }

    input.close();
}

// Copy constructor.
Network::Network(const Network &rhs) :
        _type(rhs._type),
        layers(rhs.layers)
{
    size_t num_connections = this->layers.size() - 1;
    this->connections.reserve(num_connections);
    for (size_t i = 0; i < num_connections; ++i) {
        this->connections.push_back(Connection(&this->layers[i], &this->layers[i + 1]));
        this->connections[i].weights = rhs.connections[i].weights;
        this->connections[i].bias = rhs.connections[i].bias;
    }
}

// Move constructor.
Network::Network(Network &&rhs) :
        _type(std::move(rhs._type)),
        layers(std::move(rhs.layers))
{
    size_t num_connections = this->layers.size() - 1;
    this->connections.reserve(num_connections);
    for (size_t i = 0; i < num_connections; ++i) {
        this->connections.push_back(Connection(&this->layers[i], &this->layers[i + 1]));
        this->connections[i].weights = std::move(rhs.connections[i].weights);
        this->connections[i].bias = std::move(rhs.connections[i].bias);
    }
}

// Copy assignment operator.
Network& Network::operator=(const Network &rhs) {
    this->_type = rhs._type;
    this->layers = rhs.layers;
    size_t num_connections = this->layers.size() - 1;
    this->connections.clear();
    this->connections.reserve(num_connections);
    for (size_t i = 0; i < num_connections; ++i) {
        this->connections.push_back(Connection(&this->layers[i], &this->layers[i + 1]));
        this->connections[i].weights = rhs.connections[i].weights;
        this->connections[i].bias = rhs.connections[i].bias;
    }
    return *this;
}

// Move assignment operator.
Network& Network::operator=(Network &&rhs) {
    this->_type = std::move(rhs._type);
    this->layers = std::move(rhs.layers);
    size_t num_connections = this->layers.size() - 1;
    this->connections.clear();
    this->connections.reserve(num_connections);
    for (size_t i = 0; i < num_connections; ++i) {
        this->connections.push_back(Connection(&this->layers[i], &this->layers[i + 1]));
        this->connections[i].weights = std::move(rhs.connections[i].weights);
        this->connections[i].bias = std::move(rhs.connections[i].bias);
    }
    return *this;
}

// Returns immutable member '_type'.
NetworkType Network::type() noexcept {
    return this->_type;
}

// Forward propagates data through network.
void Network::feed_forward(const Matrix &data) {
    if (data.cols() != this->layers[0].data.cols()) {
        throw invalid_argument("Data has too few or too many features!");
    }
    this->layers[0].data = data;
    for (size_t i = 0; i < this->connections.size(); ++i) {
        Matrix temp = this->layers[i].data * this->connections[i].weights;
        temp = add_to_each_row(temp, this->connections[i].bias);
        this->connections[i].to()->data = std::move(temp);
        this->connections[i].to()->activate_layer();
    }
}

// Returns prediction of given data. User can request one-hot encoding so long as the network's type is classification.
Matrix Network::predict(bool one_hot_encoding) {
    if (one_hot_encoding && this->_type != CLASSIFICATION) {
        throw invalid_argument("Network's type is classification yet one-hot encoding was requested!");
    }
    if (one_hot_encoding) {
        Matrix prediction = this->layers[this->layers.size() - 1].data;

        // If binary classification network, allow '{0}' to be a valid output for the respective data point.
        if (prediction.cols() == 1) {
            for (size_t i = 0; i < prediction.rows(); ++i) {
                prediction[i][0] = round(prediction[i][0]);
            }
        } else {
            // Iterate backwards so that if the max value occurs more than once, the element at the smallest index will be set to 1.
            for (size_t i = 0; i < prediction.rows(); ++i) {
                double index_of_max = 0;
                for (size_t j = prediction.cols(); j-- > 0; ) {
                    if (prediction[i][j] >= prediction[i][index_of_max]) {
                        index_of_max = j;
                    }
                }
                prediction.set_row(i, 0);
                prediction[i][index_of_max] = 1;
            }
        }
        return prediction;
    } else {
        return this->layers[this->layers.size() - 1].data;
    }
}

// Returns accuracy between network's prediction and target based on given data in the range [0, 1].
double Network::accuracy(const Matrix &data, const Matrix &target) {
    if (data.rows() != target.rows()) {
        throw invalid_argument("Data and target have unequal row lengths!");
    } else if (target.cols() != this->layers[this->layers.size() - 1].size()) {
        throw invalid_argument("Target has too few or too many categories!");
    }
    this->feed_forward(data);

    if (this->_type == CLASSIFICATION) {
        Matrix prediction = this->predict(true);

        size_t num_correct = 0;
        for (size_t i = 0; i < target.rows(); ++i) {
            bool correct = true;
            for (size_t j = 0; j < target.cols(); ++j) {
                if (target[i][j] != prediction[i][j]) {
                    correct = false;
                }
            }
            if (correct) {
                ++num_correct;
            }
        }
        return num_correct / (double) target.rows();
    } else {
        Matrix prediction = this->predict();

        Matrix mean_y(1, target.cols());
        for (size_t j = 0; j < target.cols(); ++j) {
            double mean_column_sum = 0;
            for (size_t i = 0; i < target.rows(); ++i) {
                mean_column_sum += target[i][j] / target.rows();
            }
            mean_y[0][j] = mean_column_sum;
        }
        mean_y = -mean_y;

        Matrix numerator(1, target.cols());
        Matrix temp = (prediction - target).mul_elem(prediction - target);
        for (size_t j = 0; j < temp.cols(); ++j) {
            for (size_t i = 0; i < temp.rows(); ++i) {
                numerator[0][j] += temp[i][j];
            }
        }

        Matrix denominator(1, target.cols());
        temp = (add_to_each_row(target, mean_y)).mul_elem(add_to_each_row(target, mean_y));
        for (size_t j = 0; j < temp.cols(); ++j) {
            for (size_t i = 0; i < temp.rows(); ++i) {
                denominator[0][j] += temp[i][j];
            }
        }

        temp = numerator.div_elem(denominator);
        double average = 0;
        for (size_t j = 0; j < temp.cols(); ++j) {
            average += temp[0][j];
        }
        average /= temp.cols();
        return 1 - average;
    }
}

// Saves network to file.
void Network::save(const string &file_path) {
    ofstream output;
    output.open(file_path, ofstream::out | ofstream::trunc);
    if (!output) {
        throw invalid_argument("Unable to save network!");
    }

    // Serialize network type.
    if (this->_type == CLASSIFICATION) {
        output << "CLASSIFICATION\n";
    } else {
        output << "REGRESSION\n";
    }

    // Serialize number of layers.
    output << this->layers.size() << "\n";

    // Serialize layer sizes.
    for (size_t i = 0; i < this->layers.size(); ++i) {
        output << this->layers[i].size() << "\n";
    }

    // Serialize activation functions.
    for (size_t i = 0; i < this->layers.size() - 1; ++i) {
        output << get_name_by_activation(this->layers[i + 1].activation()) << "\n";
    }

    // Serialize weights.
    for (size_t k = 0; k < this->connections.size(); ++k) {
        Connection &connection = this->connections[k];
        for (size_t i = 0; i < connection.weights.rows(); ++i) {
            for (size_t j = 0; j < connection.weights.cols(); ++j) {
                output << connection.weights[i][j] << "\n";
            }
        }
    }

    // Serialize bias.
    for (size_t k = 0; k < this->connections.size(); ++k) {
        Connection &connection = this->connections[k];
        for (size_t j = 0; j < connection.bias.cols(); ++j) {
            output << connection.bias[0][j] << "\n";
        }
    }

    output.close();
}

// Trains network. If multiclass classification is used, it is expected for
// there to be one and only one '1' in the target output; otherwise, the
// behavior of the network is undefined.
void Network::train(const TrainingParams& params, bool verbose, ostream &verbose_output_stream) {
    this->train(params.data,
            params.target,
            params.batch_size,
            params.max_iterations,
            params.min_accuracy,
            params.learning_rate,
            params.annealing_factor,
            params.regularization_factor,
            params.momentum_factor,
            params.shuffle,
            verbose,
            verbose_output_stream);
}

// Trains network. If multiclass classification is used, it is expected for
// there to be one and only one '1' in the target output; otherwise, the
// behavior of the network is undefined.
void Network::train(Matrix data,
        Matrix target,
        size_t batch_size,
        size_t max_iterations,
        double min_accuracy,
        double learning_rate,
        double annealing_factor,
        double regularization_factor,
        double momentum_factor,
        bool shuffle,
        bool verbose,
        ostream &verbose_output_stream)
{
    if (data.cols() != this->layers[0].size()) {
        throw invalid_argument("Data has too few or too many features!");
    } else if (data.rows() != target.rows()) {
        throw invalid_argument("Data and target have unequal row lengths!");
    } else if (target.cols() != this->layers[this->layers.size() - 1].size()) {
        throw invalid_argument("Target has too few or too many classes/outputs!");
    } else if (batch_size > data.rows() || data.rows() % batch_size != 0) {
        throw invalid_argument("Data does not support given batch size!");
    } else if (max_iterations == 0) {
        throw invalid_argument("The max amount of iterations requested is zero!");
    } else if (min_accuracy < 0 || min_accuracy > 1) {
        throw invalid_argument("The minimum accuracy required to return early is not in the range [0, 1]!");
    }

    // Constants.
    const size_t num_batches = data.rows() / batch_size + (data.rows() % batch_size == 0 ? 0 : 1);
    const std::vector<Matrix> data_batches = create_batches(data, num_batches);
    const std::vector<Matrix> target_batches = create_batches(target, num_batches);

    // Reused variables ('dW' means change in weight, 'dB' means change in bias, 'err' means error).
    std::vector<Matrix> dW(this->connections.size());
    std::vector<Matrix> dW_avg(this->connections.size());
    std::vector<Matrix> dW_last(this->connections.size());
    std::vector<Matrix> dB(this->connections.size());
    std::vector<Matrix> dB_avg(this->connections.size());
    std::vector<Matrix> dB_last(this->connections.size());
    std::vector<Matrix> err(this->layers.size());
    for (size_t i = 0; i < this->connections.size(); ++i) {
        dW_avg[i] = Matrix(this->connections[i].weights.rows(), this->connections[i].weights.cols());
        dW_last[i] = Matrix(this->connections[i].weights.rows(), this->connections[i].weights.cols());
        dB_avg[i] = Matrix(1, this->connections[i].bias.cols());
        dB_last[i] = Matrix(1, this->connections[i].bias.cols());
    }

    for (size_t epoch = 1; epoch <= max_iterations; ) {
        // Shuffle training data/target while maintaining training data/target alignment.
        if (shuffle) {
            shuffle_together(data, target);
        }

        for (size_t batch = 0; batch < num_batches && epoch <= max_iterations; ++batch, ++epoch) {
            // Operate on current batch.
            size_t cur_batch_size = batch == num_batches - 1 ? (data.rows() % batch_size != 0 ? data.rows() % batch_size : batch_size) : batch_size;
            std::vector<Matrix> split_data = create_batches(data_batches[batch], cur_batch_size);
            std::vector<Matrix> split_target = create_batches(target_batches[batch], cur_batch_size);
            for (size_t training = 0; training < cur_batch_size; ++training) {
                // Current data point to train on.
                Matrix &training_data = split_data[training];
                Matrix &training_target = split_target[training];

                // Feed forward training data.
                this->feed_forward(training_data);

                // Backpropagate.
                for (size_t layer = this->layers.size() - 1; layer > 0; --layer) {
                    Layer &to = this->layers[layer];
                    Connection &connection = this->connections[layer - 1];
                    if (layer == this->layers.size() - 1) {
                        // Calculate output layer's error.
                        err[layer] = to.data;
                        for (size_t j = 0; j < err[layer].cols(); ++j) {
                            err[layer][0][j] -= training_target[0][j];
                        }

                        // Calculate dW and dB.
                        dW[layer - 1] = connection.from()->data.transpose() * err[layer];
                        dB[layer - 1] = err[layer];
                    } else {
                        // Calculate hidden layer's error.
                        Matrix temp = connection.to()->data;
                        double (*deriv)(double) = get_activation_derivative(connection.to()->activation());
                        for (size_t j = 0; j < temp.cols(); ++j) {
                            temp[0][j] = deriv(temp[0][j]);
                        }
                        err[layer] = (err[layer + 1] * this->connections[layer].weights.transpose()).mul_elem(temp);

                        // Calculate dW and dB.
                        dW[layer - 1] = connection.from()->data.transpose() * err[layer];
                        dB[layer - 1] = err[layer];
                    }
                }

                // Add example's contribution to total gradient.
                for (size_t i = 0; i < this->connections.size(); ++i) {
                    dW_avg[i] += dW[i];
                    dB_avg[i] += dB[i];
                }
            }

            // Calculate this epoch's learning rate.
            double cur_learning_rate = annealing_factor == 0 ? learning_rate : learning_rate / (1 + (epoch / annealing_factor));

            // Average out gradients and add learning rate.
            for (size_t i = 0; i < this->connections.size(); ++i) {
                dW_avg[i] *= cur_learning_rate / data.rows();
                dB_avg[i] *= cur_learning_rate / data.rows();
            }

            // Add regularization.
            for (size_t i = 0; i < this->connections.size(); ++i) {
                dW_avg[i] += this->connections[i].weights * regularization_factor;
            }

            // Add momentum.
            for (size_t i = 0; i < this->connections.size(); ++i) {
                dW_avg[i] += dW_last[i] * momentum_factor;
                dB_avg[i] += dB_last[i] * momentum_factor;
            }

            // Adjust weights and bias.
            for (size_t i = 0; i < this->connections.size(); ++i) {
                dW_avg[i] = -dW_avg[i];
                dB_avg[i] = -dB_avg[i];
                this->connections[i].weights += dW_avg[i];
                this->connections[i].bias += dB_avg[i];
            }

            // Cache weight and bias updates for momentum.
            for (size_t i = 0; i < this->connections.size(); ++i) {
                dW_last[i] = dW_avg[i];
                dB_last[i] = dB_avg[i];

                // Make positive again for next epoch.
                dW_last[i] = -dW_last[i];
                dB_last[i] = -dB_last[i];
            }

            // Zero out reusable variables.
            for (size_t i = 0; i < this->connections.size(); ++i) {
                dW_avg[i].set_all(0);
                dB_avg[i].set_all(0);
            }

            // Every 100 epochs: print loss if verbose is true, exit if accuracy is greater than or equal to minimum required accuracy.
            if (epoch % 100 == 0 || epoch == 1) {
                if (verbose) {
                    this->feed_forward(data);
                    double loss = this->loss_function(this->predict(), target, regularization_factor);
                    verbose_output_stream << "EPOCH " << epoch << ": loss is " << loss << "\n";
                }

                if (this->accuracy(data, target) >= min_accuracy) {
                    return;
                }
            }
        }
    }
}

// Loss function (cross entropy for classification, mean squared error for regression).
double Network::loss_function(const Matrix &prediction, const Matrix &actual, double regularization_factor) {
    if (prediction.rows() != actual.rows() || prediction.cols() != actual.cols()) {
        throw invalid_argument("Matrices have unequal dimensions!");
    }

    if (this->_type == CLASSIFICATION) {
        double total_err = 0;
        for (size_t i = 0; i < prediction.rows(); ++i) {
            double cur_err = 0;
            for (size_t j = 0; j < prediction.cols(); ++j) {
                cur_err += actual[i][j] * log(numeric_limits<double>::min() > prediction[i][j] ? numeric_limits<double>::min() : prediction[i][j]);
            }
            total_err += cur_err;
        }
        double reg_err = 0;
        for (size_t i = 0; i < this->connections.size(); ++i) {
            Matrix &weights = this->connections[i].weights;
            for (size_t j = 0; j < weights.rows(); ++j) {
                for (size_t k = 0; k < weights.cols(); ++k) {
                    reg_err += pow(weights[j][k], 2);
                }
            }
        }
        return ((-1.0 / actual.rows()) * total_err) + (regularization_factor * 0.5 * reg_err);
    } else {
        double total_err = 0;
        for (size_t i = 0; i < prediction.rows(); ++i) {
            double cur_err = 0;
            for (size_t j = 0; j < prediction.cols(); ++j) {
                double temp = actual[i][j] - prediction[i][j];
                cur_err += pow(temp, 2);
            }
            total_err += cur_err;
        }
        double reg_err = 0;
        for (size_t k = 0; k < this->connections.size(); ++k) {
            Matrix &weights = this->connections[k].weights;
            for (size_t i = 0; i < weights.rows(); ++i) {
                for (size_t j = 0; j < weights.cols(); ++j) {
                    reg_err += pow(weights[i][j], 2);
                }
            }
        }
        return ((0.5 / actual.rows()) * total_err) + (regularization_factor * 0.5 * reg_err);
    }
}

} // namespace yannl
