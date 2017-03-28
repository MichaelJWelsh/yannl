/*
 * Representation of a layer in a neural network.
 */
#ifndef YANNL_LAYER_HPP
#define YANNL_LAYER_HPP

#include "prereqs.hpp"
#include "activation.hpp"
#include "matrix.hpp"

namespace yannl {
namespace internal {

// Type of layer.
enum LayerType {
    INPUT,
    HIDDEN,
    OUTPUT
};

// Layer representation.
class Layer final {
public:
    // Directly accessible type.
    Matrix data;

    // Creation.
    Layer(LayerType type, std::size_t size, Activation activation);

    // Back-end accessors.
    LayerType type() noexcept;
    std::size_t size() noexcept;
    Activation activation() noexcept;

    // Other operations.
    void activate_layer();

private:
    LayerType _type;
    std::size_t _size;
    Activation _activation;
};

} // namespace internal
} // namespace yannl

#endif // YANNL_LAYER_HPP
