COMPILER=g++
FLAGS=-lyannl -Wall -Werror -pedantic-errors -std=c++11

clean:
	rm -f classification realistic_classification regression integrating_blas

classification:
	$(COMPILER) classification_example.cpp -o classification $(FLAGS)
	./classification

realistic_classification:
	$(COMPILER) realistic_classification_example.cpp -o realistic_classification $(FLAGS)
	./realistic_classification

regression:
	$(COMPILER) regression_example.cpp -o regression $(FLAGS)
	./regression

integrating_blas:
	$(COMPILER) integrating_blas_example.cpp -o integrating_blas -lblas $(FLAGS)
	./integrating_blas
