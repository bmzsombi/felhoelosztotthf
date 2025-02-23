#include <mpi.h>
#include <omp.h>
#include <iostream>
#include <complex>
#include <chrono>
#include <vector>

void compute_mandelbrot(unsigned char* data, int startY, int endY, int width, int height) {
    std::complex<double> center(-1.68, -1.23);
    double scale = 2.35;
    const unsigned int maxIterations = 500;

    #pragma omp parallel for schedule(dynamic)
    for (int y = startY; y < endY; ++y) {
        for (int x = 0; x < width; ++x) {
            std::complex<double> c(x / (double)width * scale + center.real(),
                                   y / (double)height * scale + center.imag());
            std::complex<double> z(c);
            for (unsigned int iteration = 0; iteration < maxIterations; ++iteration) {
                z = z * z + c;
                if (std::abs(z) > 1.0f) {
                    data[(x + (y - startY) * width) * 3 + 0] = 255;
                    data[(x + (y - startY) * width) * 3 + 1] = 255;
                    data[(x + (y - startY) * width) * 3 + 2] = 255;
                    break;
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    const unsigned int width = 2048;
    const unsigned int height = 2048;
    const int rowsPerProc = height / numProcs;
    int startY = rank * rowsPerProc;
    int endY = (rank == numProcs - 1) ? height : startY + rowsPerProc;

    std::vector<unsigned char> localData(rowsPerProc * width * 3, 0);

    auto startTime = std::chrono::high_resolution_clock::now();

    compute_mandelbrot(localData.data(), startY, endY, width, height);

    std::vector<unsigned char> fullImage;
    if (rank == 0) {
        fullImage.resize(width * height * 3);
    }

    MPI_Gather(localData.data(), localData.size(), MPI_UNSIGNED_CHAR,
               fullImage.data(), localData.size(), MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    auto endTime = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() * 1e-3;
        std::cout << "Elapsed time: " << elapsed << "s" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
