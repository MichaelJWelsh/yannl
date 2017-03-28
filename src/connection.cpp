/*
 * Representation of a connection between layers of a neural network.
 */
#include "connection.hpp"

using std::numeric_limits;
using std::invalid_argument;
using std::cos;
using std::sin;
using std::log;
using std::sqrt;
using std::rand;

namespace yannl {
namespace internal {

// Creates connection.
Connection::Connection(Layer *from, Layer *to) :
        weights(from->size(), to->size()),
        bias(1, to->size()),
        _from(from),
        _to(to)
{
    for (size_t i = 0; i < this->weights.rows(); ++i) {
        for (size_t j = 0; j < this->weights.cols(); ++j) {
            this->weights[i][j] = box_muller_transform() / sqrt(this->weights.rows());
        }
    }
}

// Returns immutable member '_from'.
Layer* Connection::from() noexcept {
    return this->_from;
}

// Returns immutable member '_to'.
Layer* Connection::to() noexcept {
    return this->_to;
}

// Essentially, a modified pseudo-random number generator.
double box_muller_transform() noexcept {
    const double epsilon = numeric_limits<double>::min();
    const double two_pi = 2 * 3.14159265358979323846;

    static double z0, z1;
    static bool generate;
    generate = !generate;

    if (!generate) {
        return z1;
    }

    double u1, u2;
    do {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);

    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0;
}

} // namespace internal
} // namespace yannl
