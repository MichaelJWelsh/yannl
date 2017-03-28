#include <yannl.hpp>
#include <cblas.h>

#include "print_matrix.hpp"

using namespace yannl;

Matrix multiplication_callback (const Matrix &A, const Matrix &B) {
    Matrix result(A.rows(), B.cols());
    cblas_dgemm(CblasColMajor,
            CblasNoTrans,
            CblasNoTrans,
            A.rows(),
            B.cols(),
            A.cols(),
            1,
            const_cast<Matrix&>(A).data(),
            A.rows(),
            const_cast<Matrix&>(B).data(),
            B.rows(),
            0,
            result.data(),
            result.rows());
    return result;
}

int main() {
    Matrix::set_mul_callback(multiplication_callback);

    Matrix A = {
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9}
    };

    Matrix B = {
        {10, 20},
        {30, 40},
        {50, 60}
    };

    print_matrix(A * B);
    print_divider();

    return 0;
}
