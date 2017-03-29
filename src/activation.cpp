/*
 * Activation functions.
 */
#include "activation.hpp"

using std::string;

namespace yannl {

// Sigmoid activation function.
void sigmoid(Matrix &input) {
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input[i][j] = internal::raw_sigmoid(input[i][j]);
        }
    }
}

// Relu activation function.
void relu(Matrix &input) {
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input[i][j] = internal::raw_relu(input[i][j]);
        }
    }
}

// Tanh activation function.
void tanh(Matrix &input) {
    for (size_t i = 0; i < input.rows(); ++i) {
        for (size_t j = 0; j < input.cols(); ++j) {
            input[i][j] = internal::raw_tanh(input[i][j]);
        }
    }
}

// Softmax activation function.
void softmax(Matrix &input) {
    if (input.cols() == 1) {
        for (size_t i = 0; i < input.rows(); ++i) {
            if (input[i][0] > 1) {
                input[i][0] = 1;
            } else if (input[i][0] < 0) {
                input[i][0] = 0;
            }
        }
    } else {
        for (size_t i = 0; i < input.rows(); ++i) {
            double sum = 0;
            for (size_t j = 0; j < input.cols(); ++j) {
                sum += std::exp(input[i][j]);
            }
            for (size_t j = 0; j < input.cols(); ++j) {
                input[i][j] = std::exp(input[i][j]) / sum;
            }
        }
    }
}

// Linear activation function.
void linear(Matrix &input) { }

namespace internal {

// Raw sigmoid function.
double raw_sigmoid(double input) noexcept {
    return 1 / (1 + std::exp(-input));
}

// Raw relu function.
double raw_relu(double input) noexcept {
    return 0 > input ? 0 : input;
}

// Raw tanh function.
double raw_tanh(double input) noexcept {
    return std::tanh(input);
}

// Raw sigmoid derivative function.
double raw_sigmoid_deriv(double input) noexcept {
    return input * (1 - input);
}

// Raw relu derivative function.
double raw_relu_deriv(double input) noexcept {
    return input > 0 ? 1 : 0;
}

// Raw tanh derivative function.
double raw_tanh_deriv(double input) noexcept {
    return 1 - (input * input);
}

// Raw linear derivative function.
double raw_linear_deriv(double input) noexcept {
    return 1;
}

// Gets activation function's raw derivative function.
double (*get_activation_derivative(Activation func) noexcept)(double) {
    if (func == sigmoid) {
        return raw_sigmoid_deriv;
    } else if (func == relu) {
        return raw_relu_deriv;
    } else if (func == tanh) {
        return raw_tanh_deriv;
    } else {
        return raw_linear_deriv;
    }
}

// Returns activation function based on string.
Activation get_activation_by_name(const string &name) noexcept {
    if (name == "sigmoid") {
        return sigmoid;
    } else if (name == "relu") {
        return relu;
    } else if (name == "tanh") {
        return tanh;
    } else if (name == "softmax") {
        return softmax;
    } else {
        return linear;
    }
}

// Returns string based on activation function.
string get_name_by_activation(Activation activation) noexcept {
    if (activation == sigmoid) {
        return "sigmoid";
    } else if (activation == relu) {
        return "relu";
    } else if (activation == tanh) {
        return "tanh";
    } else if (activation == softmax) {
        return "softmax";
    } else {
        return "linear";
    }
}

} // namespace internal
} // namespace yannl
