/*
 * Representation of a connection between layers of a neural network.
 */
#ifndef YANNL_CONNECTION_HPP
#define YANNL_CONNECTION_HPP

#include "prereqs.hpp"
#include "layer.hpp"
#include "matrix.hpp"

namespace yannl {
namespace internal {

// Connection representation.
class Connection final {
public:
    // Directly accessible types.
    Matrix weights;
    Matrix bias;

    // Creation.
    Connection(Layer *from, Layer *to);

    // Back-end accessors.
    Layer* from() noexcept;
    Layer* to() noexcept;

private:
    Layer *_from;
    Layer *_to;
};

// Other operations.
double box_muller_transform() noexcept;

} // namespace internal
} // namespace yannl

#endif // YANNL_CONNECTION_HPP
