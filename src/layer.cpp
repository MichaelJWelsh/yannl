/*
 * Representation of a layer in a neural network.
 */
#include "layer.hpp"

using std::invalid_argument;

namespace yannl {
namespace internal {

// Creates layer.
Layer::Layer(LayerType type, size_t size, Activation activation) :
        data(1, size),
        _type(type),
        _size(size),
        _activation(activation)
{
    if (size == 0) {
        throw invalid_argument("Cannot create layer with zero neurons!");
    } else if (type != INPUT && activation == nullptr) {
        throw invalid_argument("Layer must have callable activation function!");
    }
}

// Returns immutable member '_type'.
LayerType Layer::type() noexcept {
    return this->_type;
}

// Returns immutable member '_size'.
size_t Layer::size() noexcept {
    return this->_size;
}

// Returns immutable member '_activation'.
Activation Layer::activation() noexcept {
    return this->_activation;
}

// Activates layer.
void Layer::activate_layer() {
    if (this->_activation != nullptr) {
        this->_activation(this->data);
    }
}

} // namespace internal
} // namespace yannl
