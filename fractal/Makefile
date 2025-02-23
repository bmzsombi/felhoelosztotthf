# 
# Example Makefile to compile and run
# should be modified according to the tools of the paralellization
#
CXX = mpic++
CXXFLAGS = -Wall -fopenmp
LIBS = 


# compile
all: fractal.cpp
	$(CXX) $(CXXFLAGS) -o fractal fractal.cpp $(LIBS)

# run on the cluster
.phony:
run:	fractal
	srun --mpi=pmi2 -o fractal.out -N 2 -n 8 ./fractal
	cat fractal.out

# run on the cluster
.phony:
batch:	fractal
	sbatch fractal.sh

.phony:
clean:
	rm -rf fractal mandelbrot.tga *.out


