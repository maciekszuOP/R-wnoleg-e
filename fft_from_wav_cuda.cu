#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cuda_runtime.h>

using namespace std;

#define PI 3.14159265358979323846
//nvcc -o fft_from_wav_cuda.cu main.cu -I/path/to/dr_wav -ldl

struct Complex {
    double real;
    double imag;

    __host__ __device__ Complex(double r = 0.0, double i = 0.0) : real(r), imag(i) {}

    __host__ __device__ Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    __host__ __device__ Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    __host__ __device__ Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    __host__ __device__ double abs() const {
        return sqrt(real * real + imag * imag);
    }
};

__global__ void fft_stage(Complex* data, int n, int stage) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int m = 1 << stage;
    int half_m = m / 2;
    if (tid >= n / 2) return;

    int k = (tid / half_m) * m;
    int j = tid % half_m;

    double angle = -2.0 * PI * j / m;
    Complex wm(cos(angle), sin(angle));

    Complex t = wm * data[k + j + half_m];
    Complex u = data[k + j];
    data[k + j] = u + t;
    data[k + j + half_m] = u - t;
}

void bit_reversal(vector<Complex>& a) {
    int n = a.size();
    int log_n = static_cast<int>(log2(n));
    vector<Complex> a_reordered(n);
    for (int i = 0; i < n; ++i) {
        int j = 0;
        for (int bit = 0; bit < log_n; ++bit) {
            if (i & (1 << bit))
                j |= 1 << (log_n - 1 - bit);
        }
        a_reordered[j] = a[i];
    }
    a = a_reordered;
}

vector<Complex> loadWavAsComplex(const char* filename) {
    unsigned int channels;
    unsigned int sampleRate;
    drwav_uint64 totalFrameCount;

    float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
        filename, &channels, &sampleRate, &totalFrameCount, nullptr);

    if (pSampleData == nullptr) {
        throw runtime_error("Failed to load WAV file.");
    }

    vector<Complex> result;
    for (drwav_uint64 i = 0; i < totalFrameCount * channels; i += channels) {
        float mono = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            mono += pSampleData[i + c];
        }
        mono /= channels;
        result.emplace_back(mono, 0.0);
    }

    drwav_free(pSampleData, nullptr);
    return result;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename.wav>\n";
        return 1;
    }

    const char* filename = argv[1];

    vector<Complex> data;
    try {
        data = loadWavAsComplex(filename);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    int N = 1 << static_cast<int>(log2(data.size()));
    data.resize(N);
    cout << "Loaded " << N << " samples from " << filename << endl;

    bit_reversal(data);

    Complex* d_data;
    cudaMalloc((void**)&d_data, sizeof(Complex) * N);
    cudaMemcpy(d_data, data.data(), sizeof(Complex) * N, cudaMemcpyHostToDevice);

    auto start = chrono::high_resolution_clock::now();

    int threadsPerBlock = 256;
    for (int s = 1; s <= static_cast<int>(log2(N)); ++s) {
        int blocks = (N / 2 + threadsPerBlock - 1) / threadsPerBlock;
        fft_stage<<<blocks, threadsPerBlock>>>(d_data, N, s);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(data.data(), d_data, sizeof(Complex) * N, cudaMemcpyDeviceToHost);

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;

    cout << "FFT completed in " << duration.count() << " ms\n";

    for (int i = 0; i < 10; ++i) {
        cout << i << ": " << data[i].abs() << "\n";
    }

    cudaFree(d_data);
    return 0;
}
