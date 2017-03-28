#include <yannl.hpp>
#include <cstdio>

using std::printf;

inline void print_matrix(const yannl::Matrix &m) {
    printf("<--MATRIX-->\nRows:%zd\nColumns:%zd\n\t\t", m.rows(), m.cols());
    for (size_t i = 0; i < m.rows(); ++i) {
        for (size_t j = 0; j < m.cols(); ++j) {
            printf("%7.1f", m[i][j]);
        }
        printf("\n\n\t\t");
    }
    printf("\n");
}

inline void print_divider() {
    printf("---------------------------------------------------\n");
}
