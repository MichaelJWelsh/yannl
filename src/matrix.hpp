/*
 * The Yannl type for storing and manipulating data (stored in column-major order).
 */
#ifndef YANNL_MATRIX_HPP
#define YANNL_MATRIX_HPP

#include "prereqs.hpp"

namespace yannl {

// Matrix representation.
class Matrix final {
public:
    // Used to refer to an optimized multiplication callback if supplied by the user.
    typedef Matrix (*MulCallback)(const Matrix &, const Matrix &);

    // Used as a medium for matrix[r][c] access.
    class SubscriptMedium final {
    public:
        double& operator[](std::size_t col);
        const double& operator[](std::size_t col) const;

    private:
        Matrix *_self;
        std::size_t _row;

        SubscriptMedium(const Matrix &self, std::size_t row);

        friend Matrix;
    };

    // Optional external optimization functionality.
    static void set_mul_callback(MulCallback mul_callback) noexcept;
    static MulCallback get_mul_callback() noexcept;

    // Creation.
    Matrix();
    Matrix(std::initializer_list<std::initializer_list<double>> data);
    Matrix(std::size_t rows, std::size_t cols, double default_val = 0);

    // Back-end accessors.
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    double* data() noexcept;
    const double* data() const noexcept;

    // Front-end get/sets.
    double get(std::size_t row, std::size_t col) const;
    Matrix& set(std::size_t row, std::size_t col, double data);
    Matrix& set_all(double data) noexcept;
    Matrix& set_row(std::size_t row, double data);
    Matrix& set_col(std::size_t col, double data);
    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row, std::size_t col) const;
    SubscriptMedium operator[](std::size_t row);
    const SubscriptMedium operator[](std::size_t row) const;

    // Physical manipulation.
    Matrix& cpy_row(std::size_t dest_row, std::size_t src_row);
    Matrix& cpy_row(std::size_t dest_row, std::size_t src_row, const Matrix &src);
    Matrix& cpy_col(std::size_t dest_col, std::size_t src_col);
    Matrix& cpy_col(std::size_t dest_col, std::size_t src_col, const Matrix &src);
    Matrix& cpy_elem(std::size_t dest_row, std::size_t dest_col, std::size_t src_row, std::size_t src_col);
    Matrix& cpy_elem(std::size_t dest_row, std::size_t dest_col, std::size_t src_row, std::size_t src_col, const Matrix &src);
    Matrix& swap_row(std::size_t row, std::size_t other_row);
    Matrix& swap_row(std::size_t row, std::size_t other_row, Matrix &other);
    Matrix& swap_col(std::size_t col, std::size_t other_col);
    Matrix& swap_col(std::size_t col, std::size_t other_col, Matrix &other);
    Matrix& swap_elem(std::size_t row, std::size_t col, std::size_t other_row, std::size_t other_col);
    Matrix& swap_elem(std::size_t row, std::size_t col, std::size_t other_row, std::size_t other_col, Matrix &other);
    Matrix& ins_row(std::size_t dest_row, std::size_t src_row);
    Matrix& ins_row(std::size_t dest_row, std::size_t src_row, const Matrix &src);
    Matrix& ins_col(std::size_t dest_col, std::size_t src_col);
    Matrix& ins_col(std::size_t dest_col, std::size_t src_col, const Matrix &src);
    Matrix& del_all() noexcept;
    Matrix& del_row(std::size_t row);
    Matrix& del_col(std::size_t col);
    Matrix& adjoin_top(const Matrix &src);
    Matrix& adjoin_bottom(const Matrix &src);
    Matrix& adjoin_left(const Matrix &src);
    Matrix& adjoin_right(const Matrix &src);

    // Arithmetic.
    Matrix add_scalar(double rhs) const;
    Matrix mul_scalar(double rhs) const;
    Matrix add(const Matrix &rhs) const;
    Matrix mul(const Matrix &rhs) const;
    Matrix mul_elem(const Matrix &rhs) const;
    Matrix div_elem(const Matrix &rhs) const;
    Matrix& comp_add_scalar(double rhs);
    Matrix& comp_mul_scalar(double rhs);
    Matrix& comp_add(const Matrix &rhs);
    Matrix& comp_mul(const Matrix &rhs);
    Matrix& comp_mul_elem(const Matrix &rhs);
    Matrix& comp_div_elem(const Matrix &rhs);

    // Other operations.
    Matrix transpose() const;
    Matrix& comp_transpose();
    Matrix normalize_rescale() const;
    Matrix normalize_standardize() const;
    Matrix& comp_normalize_rescale();
    Matrix& comp_normalize_standardize();
    double min() const;
    double max() const;
    bool is_zero() const;
    bool is_pos() const;
    bool is_neg() const;
    bool is_nonpos() const;
    bool is_nonneg() const;
    bool is_equal(const Matrix &other) const noexcept;

private:
    std::size_t _rows;
    std::size_t _cols;
    std::vector<double> _data;
    static MulCallback _mul_callback;

    double& access_elem_via_ref(std::size_t row, std::size_t col);
    const double& access_elem_via_ref(std::size_t row, std::size_t col) const;
};

// Friends and operator overloading.
Matrix operator+(const Matrix &lhs, double rhs);
Matrix operator+(double lhs, const Matrix &rhs);
Matrix operator+(const Matrix &lhs, const Matrix &rhs);
Matrix operator+(const Matrix &rhs);
Matrix operator-(const Matrix &lhs, double rhs);
Matrix operator-(double lhs, const Matrix &rhs);
Matrix operator-(const Matrix &lhs, const Matrix &rhs);
Matrix operator-(const Matrix &rhs);
Matrix operator*(const Matrix &lhs, double rhs);
Matrix operator*(double lhs, const Matrix &rhs);
Matrix operator*(const Matrix &lhs, const Matrix &rhs);
Matrix operator/(const Matrix &lhs, double rhs);
Matrix& operator+=(Matrix &lhs, double rhs);
Matrix& operator+=(Matrix &lhs, const Matrix &rhs);
Matrix& operator-=(Matrix &lhs, double rhs);
Matrix& operator-=(Matrix &lhs, const Matrix &rhs);
Matrix& operator*=(Matrix &lhs, double rhs);
Matrix& operator*=(Matrix &lhs, const Matrix &rhs);
Matrix& operator/=(Matrix &lhs, double rhs);
bool operator==(const Matrix &lhs, const Matrix &rhs) noexcept;
bool operator!=(const Matrix &lhs, const Matrix &rhs) noexcept;

namespace internal {

// Other operations.
Matrix add_to_each_row(const Matrix &data, const Matrix &row_vec);
void shuffle_together(Matrix &data, Matrix &target);
std::vector<Matrix> create_batches(const Matrix &data, size_t num_batches);

} // namespace internal
} // namespace yannl

#endif // YANNL_MATRIX_HPP
