/*
 * The Yannl type for storing and manipulating data (stored in column-major order).
 */
#include "matrix.hpp"

using std::initializer_list;
using std::size_t;
using std::invalid_argument;
using std::domain_error;
using std::pow;
using std::sqrt;
using std::rand;

namespace yannl {

// Initialize static members with a default value.
Matrix::MulCallback Matrix::_mul_callback = nullptr;

// Used for matrix[r][c] access.
double& Matrix::SubscriptMedium::operator[](size_t col) {
    return this->_self->access_elem_via_ref(this->_row, col);
}

// Used for matrix[r][c] access.
const double& Matrix::SubscriptMedium::operator[](size_t col) const {
    return this->_self->access_elem_via_ref(this->_row, col);
}

// Creates a SubscriptMedium. The const_cast is GUARANTEED to be safe.
Matrix::SubscriptMedium::SubscriptMedium(const Matrix &self, size_t row) :
        _self(const_cast<Matrix*>(&self)),
        _row(row)
{ }

// Sets multiplication callback.
void Matrix::set_mul_callback(Matrix::MulCallback mul_callback) noexcept {
    Matrix::_mul_callback = mul_callback;
}

// Gets multiplication callback.
Matrix::MulCallback Matrix::get_mul_callback() noexcept {
    return Matrix::_mul_callback;
}

// Creates dimensionless matrix.
Matrix::Matrix() :
        Matrix(0, 0)
{ }

// Creates matrix from 2-dimensional initializer_list.
Matrix::Matrix(initializer_list<initializer_list<double>> data) :
        Matrix(data.size(), data.size() == 0 ? 0 : data.begin()->size())
{
    for (size_t i = 0; i < this->_rows; ++i) {
        if (data.begin()[i].size() != this->_cols) {
            throw invalid_argument("Given data is jagged!");
        }
        for (size_t j = 0; j < this->_cols; ++j) {
            (*this)[i][j] = data.begin()[i].begin()[j];
        }
    }
}

// Creates matrix.
Matrix::Matrix(size_t rows, size_t cols, double default_val) :
        _rows(rows),
        _cols(cols),
        _data(std::vector<double>(rows * cols, default_val))
{
    if ((rows == 0 && cols > 0) || (cols == 0 && rows > 0)) {
        throw invalid_argument("Matrix must be either entirely dimensionless or at least a 1x1!");
    }
}

// Returns immutable member '_rows'.
size_t Matrix::rows() const noexcept {
    return this->_rows;
}

// Returns immutable member '_cols'.
size_t Matrix::cols() const noexcept {
    return this->_cols;
}

// Returns pointer to COLUMN-MAJOR vector of data.
double* Matrix::data() noexcept {
    return this->_data.data();
}

// Returns pointer to COLUMN-MAJOR vector of data.
const double* Matrix::data() const noexcept {
    return this->_data.data();
}

// Returns rvalue of element at specified index.
double Matrix::get(size_t row, size_t col) const {
    return (*this)[row][col];
}

// Sets element at specified index.
Matrix& Matrix::set(size_t row, size_t col, double data) {
    (*this)[row][col] = data;
    return *this;
}

// Sets all elements.
Matrix& Matrix::set_all(double data) noexcept {
    for (size_t i = 0; i < this->_data.size(); ++i) {
        this->_data[i] = data;
    }
    return *this;
}

// Sets all elements in specified row.
Matrix& Matrix::set_row(size_t row, double data) {
    for (size_t j = 0; j < this->_cols; ++j) {
        (*this)[row][j] = data;
    }
    return *this;
}

// Sets all elements in specified column.
Matrix& Matrix::set_col(size_t col, double data) {
    for (size_t i = 0; i < this->_rows; ++i) {
        (*this)[i][col] = data;
    }
    return *this;
}

// Allows matrix(r, c) access (to feel a bit like matlab).
double& Matrix::operator()(size_t row, size_t col) {
    return (*this)[row][col];
}

// Allows matrix(r, c) access (to feel a bit like matlab).
const double& Matrix::operator()(size_t row, size_t col) const {
    return (*this)[row][col];
}

// Used for matrix[r][c] access.
Matrix::SubscriptMedium Matrix::operator[](size_t row) {
    return Matrix::SubscriptMedium(*this, row);
}

// Used for matrix[r][c] access.
const Matrix::SubscriptMedium Matrix::operator[](size_t row) const {
    return Matrix::SubscriptMedium(*this, row);
}

// Copy row.
Matrix& Matrix::cpy_row(size_t dest_row, size_t src_row) {
    return this->cpy_row(dest_row, src_row, *this);
}

// Copy row from source.
Matrix& Matrix::cpy_row(size_t dest_row, size_t src_row, const Matrix &src) {
    if (this->_cols != src._cols) {
        throw invalid_argument("Unequal column lengths!");
    }
    for (size_t j = 0; j < this->_cols; ++j) {
        (*this)[dest_row][j] = src[src_row][j];
    }
    return *this;
}

// Copy column.
Matrix& Matrix::cpy_col(size_t dest_col, size_t src_col) {
    return this->cpy_col(dest_col, src_col, *this);
}

// Copy column from source.
Matrix& Matrix::cpy_col(size_t dest_col, size_t src_col, const Matrix &src) {
    if (this->_rows != src._rows) {
        throw invalid_argument("Unequal row lengths!");
    }
    for (size_t i = 0; i < this->_rows; ++i) {
        (*this)[i][dest_col] = src[i][src_col];
    }
    return *this;
}

// Copy element.
Matrix& Matrix::cpy_elem(size_t dest_row, size_t dest_col, size_t src_row, size_t src_col) {
    return this->cpy_elem(dest_row, dest_col, src_row, src_col, *this);
}

// Copy element from source.
Matrix& Matrix::cpy_elem(size_t dest_row, size_t dest_col, size_t src_row, size_t src_col, const Matrix &src) {
    (*this)[dest_row][dest_col] = src[src_row][src_col];
    return *this;
}

// Swap rows.
Matrix& Matrix::swap_row(size_t row, size_t other_row) {
    return this->swap_row(row, other_row, *this);
}

// Swap rows with other matrix.
Matrix& Matrix::swap_row(size_t row, size_t other_row, Matrix &other) {
    if (this->_cols != other._cols) {
        throw invalid_argument("Unequal column lengths!");
    }
    for (size_t j = 0; j < this->_cols; ++j) {
        std::swap((*this)[row][j], other[other_row][j]);
    }
    return *this;
}

// Swap columns.
Matrix& Matrix::swap_col(size_t col, size_t other_col) {
    return this->swap_col(col, other_col, *this);
}

// Swap columns with other matrix.
Matrix& Matrix::swap_col(size_t col, size_t other_col, Matrix &other) {
    if (this->_rows != other._rows) {
        throw invalid_argument("Unequal row lengths!");
    }
    for (size_t i = 0; i < this->_rows; ++i) {
        std::swap((*this)[i][col], other[i][other_col]);
    }
    return *this;
}

// Swap elements.
Matrix& Matrix::swap_elem(size_t row, size_t col, size_t other_row, size_t other_col) {
    return this->swap_elem(row, col, other_row, other_col, *this);
}

// Swap elements from other matrix.
Matrix& Matrix::swap_elem(size_t row, size_t col, size_t other_row, size_t other_col, Matrix &other) {
    std::swap((*this)[row][col], other[other_row][other_col]);
    return *this;
}

// Insert row.
Matrix& Matrix::ins_row(size_t dest_row, size_t src_row) {
    return this->ins_row(dest_row, src_row, *this);
}

// Insert row from source.
Matrix& Matrix::ins_row(size_t dest_row, size_t src_row, const Matrix &src) {
    if (this->_cols != src._cols && this->_cols != 0) {
        throw invalid_argument("Unequal column lengths!");
    } else if (dest_row > this->_rows) {
        throw invalid_argument("Invalid row index!");
    }
    if (this->_cols == 0) {
        this->_cols = src._cols;
    }
    this->_data.reserve(this->_data.size() + this->_cols);
    for (size_t j = this->_cols; j-- > 0; ) {
        this->_data.insert(this->_data.begin() + (this->_rows * j + dest_row), src[src_row][j]);
    }
    ++this->_rows;
    return *this;
}

// Insert column.
Matrix& Matrix::ins_col(size_t dest_col, size_t src_col) {
    return this->ins_col(dest_col, src_col, *this);
}

// Insert column from source.
Matrix& Matrix::ins_col(size_t dest_col, size_t src_col, const Matrix &src) {
    if (this->_rows != src._rows && this->_rows != 0) {
        throw invalid_argument("Unequal row lengths!");
    } else if (dest_col > this->_cols) {
        throw invalid_argument("Invalid column index!");
    }
    if (this->_rows == 0) {
        this->_rows = src._rows;
    }
    this->_data.reserve(this->_data.size() + this->_rows);
    for (size_t i = this->_rows; i-- > 0; ) {
        this->_data.insert(this->_data.begin() + (this->_rows * dest_col), src[i][src_col]);
    }
    ++this->_cols;
    return *this;
}

// Delete all elements and make matrix dimensionless.
Matrix& Matrix::del_all() noexcept {
    this->_rows = 0;
    this->_cols = 0;
    this->_data.clear();
    this->_data.shrink_to_fit();
    return *this;
}

// Delete row.
Matrix& Matrix::del_row(size_t row) {
    if (row >= this->_rows) {
        throw invalid_argument("Invalid row index!");
    }
    for (size_t j = this->_cols; j-- > 0; ) {
        this->_data.erase(this->_data.begin() + (this->_rows * j + row));
    }
    if(--this->_rows == 0) {
        this->_cols = 0;
    }
    this->_data.shrink_to_fit();
    return *this;
}

// Delete column.
Matrix& Matrix::del_col(size_t col) {
    if (col >= this->_cols) {
        throw invalid_argument("Invalid column index!");
    }
    for (size_t i = this->_rows; i-- > 0; ) {
        this->_data.erase(this->_data.begin() + (this->_rows * col));
    }
    if (--this->_cols == 0) {
        this->_rows = 0;
    }
    this->_data.shrink_to_fit();
    return *this;
}

// Stack source right on top of this matrix.
Matrix& Matrix::adjoin_top(const Matrix &src) {
    if (this->_cols != src._cols) {
        throw invalid_argument("Unequal column lengths!");
    }
    this->_data.reserve(this->_data.size() + src._data.size());
    for (size_t i = src._rows; i-- > 0; ) {
        this->ins_row(0, i, src);
    }
    return *this;
}

// Slam source to the bottom of this matrix.
Matrix& Matrix::adjoin_bottom(const Matrix &src) {
    if (this->_cols != src._cols) {
        throw invalid_argument("Unequal column lengths!");
    }
    this->_data.reserve(this->_data.size() + src._data.size());
    for (size_t i = 0; i < src._rows; ++i) {
        this->ins_row(this->_rows, i, src);
    }
    return *this;
}

// Smack source to the left of this matrix.
Matrix& Matrix::adjoin_left(const Matrix &src) {
    if (this->_rows != src._rows) {
        throw invalid_argument("Unequal row lengths!");
    }
    this->_data.reserve(this->_data.size() + src._data.size());
    for (size_t j = src._cols; j-- > 0; ) {
        this->ins_col(0, j, src);
    }
    return *this;
}

// Spank source to the right of this matrix.
Matrix& Matrix::adjoin_right(const Matrix &src) {
    if (this->_rows != src._rows) {
        throw invalid_argument("Unequal row lengths!");
    }
    this->_data.reserve(this->_data.size() + src._data.size());
    for (size_t j = 0; j < src._cols; ++j) {
        this->ins_col(this->_cols, j, src);
    }
    return *this;
}

// Add scalar.
Matrix Matrix::add_scalar(double rhs) const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    Matrix result = *this;
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] += rhs;
    }
    return result;
}

// Multiply by scalar.
Matrix Matrix::mul_scalar(double rhs) const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    Matrix result = *this;
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] *= rhs;
    }
    return result;
}

// Add matrix rhs.
Matrix Matrix::add(const Matrix &rhs) const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    } else if (this->_rows != rhs._rows || this->_cols != rhs._cols) {
        throw invalid_argument("Matrices have unequal dimensions!");
    }
    Matrix result = *this;
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] += rhs._data[i];
    }
    return result;
}

// Multiply by matrix rhs.
Matrix Matrix::mul(const Matrix &rhs) const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    } else if (this->_cols != rhs._rows) {
        throw invalid_argument("Matrices have incompatible dimensions!");
    }
    if (Matrix::_mul_callback != nullptr) {
        return Matrix::_mul_callback(*this, rhs);
    } else {
        Matrix result(this->_rows, rhs._cols);
        for (size_t i = 0; i < this->_rows; ++i) {
            for (size_t j = 0; j < rhs._cols; ++j) {
                double sum = 0;
                for (size_t k = 0; k < this->_cols; ++k) {
                    sum += (*this)[i][k] * rhs[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }
}

// Multiply each element by the corresponding rhs elements.
Matrix Matrix::mul_elem(const Matrix &rhs) const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    } else if (this->_rows != rhs._rows || this->_cols != rhs._cols) {
        throw invalid_argument("Matrices have unequal dimensions!");
    }
    Matrix result = *this;
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] *= rhs._data[i];
    }
    return result;
}

// Divide each element by the corresponding rhs elements.
Matrix Matrix::div_elem(const Matrix &rhs) const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    } else if (this->_rows != rhs._rows || this->_cols != rhs._cols) {
        throw invalid_argument("Matrices have unequal dimensions!");
    }
    Matrix result = *this;
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] /= rhs._data[i];
    }
    return result;
}

// Compound add scalar.
Matrix& Matrix::comp_add_scalar(double rhs) {
    *this = this->add_scalar(rhs);
    return *this;
}

// Compound multiply by scalar.
Matrix& Matrix::comp_mul_scalar(double rhs) {
    *this = this->mul_scalar(rhs);
    return *this;
}

// Compound add rhs matrix.
Matrix& Matrix::comp_add(const Matrix &rhs) {
    *this = this->add(rhs);
    return *this;
}

// Compound multiply by rhs matrix.
Matrix& Matrix::comp_mul(const Matrix &rhs) {
    *this = this->mul(rhs);
    return *this;
}

// Compound multiply each element by the corresponding rhs elements.
Matrix& Matrix::comp_mul_elem(const Matrix &rhs) {
    *this = this->mul_elem(rhs);
    return *this;
}

// Compound divide each element by the corresponding rhs elements.
Matrix& Matrix::comp_div_elem(const Matrix &rhs) {
    *this = this->div_elem(rhs);
    return *this;
}

// Transpose matrix.
Matrix Matrix::transpose() const {
    Matrix transpose(this->_cols, this->_rows);
    for (size_t i = 0; i < this->_rows; ++i) {
        for (size_t j = 0; j < this->_cols; ++j) {
            transpose[j][i] = (*this)[i][j];
        }
    }
    return transpose;
}

// Compound transpose matrix.
Matrix& Matrix::comp_transpose() {
    *this = this->transpose();
    return *this;
}

// Normalize matrix via rescaling.
Matrix Matrix::normalize_rescale() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    double min = this->min();
    double max = this->max();
    if (max - min == 0) {
        throw domain_error("Divide by zero!");
    }
    Matrix result = *this;
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] = (result._data[i] - min) / (max - min);
    }
    return result;
}

// Normalize matrix via standardizing.
Matrix Matrix::normalize_standardize() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    } else if (this->_data.size() == 1) {
        throw domain_error("Divide by zero!");
    }
    Matrix result = *this;
    double mean = 0;
    double standard_deviation = 0;
    for (size_t i = 0; i < result._data.size(); ++i) {
        mean += result._data[i] / result._data.size();
    }
    for (size_t i = 0; i < result._data.size(); ++i) {
        standard_deviation += pow(result._data[i] - mean, 2) / result._data.size();
    }
    standard_deviation = sqrt(standard_deviation);
    for (size_t i = 0; i < result._data.size(); ++i) {
        result._data[i] = (result._data[i] - mean) / standard_deviation;
    }
    return result;
}

// Compound normalize matrix via rescaling.
Matrix& Matrix::comp_normalize_rescale() {
    *this = this->normalize_rescale();
    return *this;
}

// Compound normalize matrix via standardizing.
Matrix& Matrix::comp_normalize_standardize() {
    *this = this->normalize_standardize();
    return *this;
}

// Returns minimum element.
double Matrix::min() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    double min = (*this)[0][0];
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] < min) {
            min = this->_data[i];
        }
    }
    return min;
}

// Returns maximum element.
double Matrix::max() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    double max = (*this)[0][0];
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] > max) {
            max = this->_data[i];
        }
    }
    return max;
}

// Returns true if matrix is entirely filled with zeros.
bool Matrix::is_zero() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] != 0) {
            return false;
        }
    }
    return true;
}

// Returns true if matrix is entirely positive.
bool Matrix::is_pos() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] <= 0) {
            return false;
        }
    }
    return true;
}

// Returns true if matrix is entirely negative.
bool Matrix::is_neg() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] >= 0) {
            return false;
        }
    }
    return true;
}

// Returns true if matrix is entirely non-positive.
bool Matrix::is_nonpos() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] > 0) {
            return false;
        }
    }
    return true;
}

// Returns true if matrix is entirely non-negative.
bool Matrix::is_nonneg() const {
    if (this->_rows == 0) {
        throw invalid_argument("Use of dimensionless matrix!");
    }
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] < 0) {
            return false;
        }
    }
    return true;
}

// Returns true if both matrices are equal in sizes and elements.
bool Matrix::is_equal(const Matrix &other) const noexcept {
    if (this->_rows != other._rows || this->_cols != other._cols) {
        return false;
    }
    for (size_t i = 0; i < this->_data.size(); ++i) {
        if (this->_data[i] != other._data[i]) {
            return false;
        }
    }
    return true;
}

// Used to directly access element.
double& Matrix::access_elem_via_ref(size_t row, size_t col) {
    if (row >= this->_rows) {
        throw invalid_argument("Invalid row index!");
    } else if (col >= this->_cols) {
        throw invalid_argument("Invalid column index!");
    }
    return this->_data[this->_rows * col + row];
}

// Used to directly access element.
const double& Matrix::access_elem_via_ref(std::size_t row, std::size_t col) const {
    if (row >= this->_rows) {
        throw invalid_argument("Invalid row index!");
    } else if (col >= this->_cols) {
        throw invalid_argument("Invalid column index!");
    }
    return this->_data[this->_rows * col + row];
}

// Binary addition overload.
Matrix operator+(const Matrix &lhs, double rhs) {
    return lhs.add_scalar(rhs);
}

// Binary addition overload.
Matrix operator+(double lhs, const Matrix &rhs) {
    return rhs.add_scalar(lhs);
}

// Binary addition overload.
Matrix operator+(const Matrix &lhs, const Matrix &rhs) {
    return lhs.add(rhs);
}

// Unary addition overload.
Matrix operator+(const Matrix &rhs) {
    return rhs;
}

// Binary subtraction overload.
Matrix operator-(const Matrix &lhs, double rhs) {
    return lhs.add_scalar(-rhs);
}

// Binary subtraction overload.
Matrix operator-(double lhs, const Matrix &rhs) {
    return (-rhs).add_scalar(lhs);
}

// Binary subtraction overload.
Matrix operator-(const Matrix &lhs, const Matrix &rhs) {
    return lhs.add(-rhs);
}

// Unary subtraction overload.
Matrix operator-(const Matrix &rhs) {
    return rhs.mul_scalar(-1);
}

// Binary multiplication overload.
Matrix operator*(const Matrix &lhs, double rhs) {
    return lhs.mul_scalar(rhs);
}

// Binary multiplication overload.
Matrix operator*(double lhs, const Matrix &rhs) {
    return rhs.mul_scalar(lhs);
}

// Binary multiplication overload.
Matrix operator*(const Matrix &lhs, const Matrix &rhs) {
    return lhs.mul(rhs);
}

// Binary division overload.
Matrix operator/(const Matrix &lhs, double rhs) {
    return lhs.mul_scalar(1 / rhs);
}

// Compound addition overload.
Matrix& operator+=(Matrix &lhs, double rhs) {
    return lhs.comp_add_scalar(rhs);
}

// Compound addition overload.
Matrix& operator+=(Matrix &lhs, const Matrix &rhs) {
    return lhs.comp_add(rhs);
}

// Compound subtraction overload.
Matrix& operator-=(Matrix &lhs, double rhs) {
    return lhs.comp_add_scalar(-rhs);
}

// Compound subtraction overload.
Matrix& operator-=(Matrix &lhs, const Matrix &rhs) {
    return lhs.comp_add(-rhs);
}

// Compound multiplication overload.
Matrix& operator*=(Matrix &lhs, double rhs) {
    return lhs.comp_mul_scalar(rhs);
}

// Compound multiplication overload.
Matrix& operator*=(Matrix &lhs, const Matrix &rhs) {
    return lhs.comp_mul(rhs);
}

// Compound division overload.
Matrix& operator/=(Matrix &lhs, double rhs) {
    return lhs.comp_mul_scalar(1 / rhs);
}

// Equality overload.
bool operator==(const Matrix &lhs, const Matrix &rhs) noexcept {
    return lhs.is_equal(rhs);
}

// Inequality overload.
bool operator!=(const Matrix &lhs, const Matrix &rhs) noexcept {
    return !(lhs == rhs);
}

namespace internal {

// Adds row_vec to each row of data.
Matrix add_to_each_row(const Matrix &data, const Matrix &row_vec) {
    Matrix result = data;
    for (size_t i = 0; i < data.rows(); ++i) {
        for (size_t j = 0; j < data.cols(); ++j) {
            result[i][j] = data[i][j] + row_vec[0][j];
        }
    }
    return result;
}

// Shuffles two matrices, maintaining alignment between rows.
void shuffle_together(Matrix &data, Matrix &target) {
    for (size_t i = 0; i < data.rows() - 1; ++i) {
        size_t j = i + rand() / (RAND_MAX / (data.rows() - i) + 1);
        data.swap_row(i, j);
        target.swap_row(i, j);
    }
}

// Splits data into batches.
std::vector<Matrix> create_batches(const Matrix &data, size_t num_batches) {
    std::vector<Matrix> batches(num_batches);
    long long remainder = data.rows() % num_batches;
    size_t cur_row = 0;
    for (size_t k = 0; k < num_batches; ++k) {
        size_t batch_size = data.rows() / num_batches;
        if (remainder-- > 0) {
            ++batch_size;
        }
        batches[k] = Matrix(batch_size, data.cols());
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < data.cols(); ++j) {
                batches[k][i][j] = data[cur_row + i][j];
            }
        }
        cur_row += batch_size;
    }
    return batches;
}

} // namespace internal
} // namespace yannl
