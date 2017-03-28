/*
 * Activation type and functions.
 */
#ifndef YANNL_ACTIVATION_HPP
#define YANNL_ACTIVATION_HPP

#include "prereqs.hpp"
#include "matrix.hpp"

namespace yannl {

// Used to refer to activation functions.
typedef void (*Activation)(Matrix&);

// Activation functions.
void sigmoid(Matrix &input);
void relu(Matrix &input);
void tanh(Matrix &input);
void softmax(Matrix &input);
void linear(Matrix &input);

namespace internal {

// Other operations.
double raw_sigmoid(double input) noexcept;
double raw_relu(double input) noexcept;
double raw_tanh(double input) noexcept;
double raw_sigmoid_deriv(double input) noexcept;
double raw_relu_deriv(double input) noexcept;
double raw_tanh_deriv(double input) noexcept;
double raw_linear_deriv(double input) noexcept;
double (*get_activation_derivative(Activation func) noexcept)(double);
Activation get_activation_by_name(const std::string &name) noexcept;
std::string get_name_by_activation(Activation activation) noexcept;

} // namespace internal
} // namespace yannl

#endif // YANNL_ACTIVATION_HPP
