#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <cufft.h>  // cuFFT
#include <cuda_runtime.h>

using namespace std;

vector<cufftDoubleComplex> loadWavAsComplex(const char* filename) {
    unsigned int channels;
    unsigned int sampleRate;
    drwav_uint64 totalFrameCount;

    float* pSampleData = drwav_open_file_and_read_pcm_frames_f32(
        filename, &channels, &sampleRate, &totalFrameCount, nullptr);

    if (pSampleData == nullptr) {
        throw runtime_error("Failed to load WAV file.");
    }

    vector<cufftDoubleComplex> result;
    for (drwav_uint64 i = 0; i < totalFrameCount * channels; i += channels) {
        float mono = 0.0f;
        for (unsigned int c = 0; c < channels; ++c) {
            mono += pSampleData[i + c];
        }
        mono /= channels;
        result.push_back({ static_cast<double>(mono), 0.0 });
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
    vector<cufftDoubleComplex> data;

    try {
        data = loadWavAsComplex(filename);
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    int N = 1 << static_cast<int>(log2(data.size()));  // next power of 2
    data.resize(N); // truncate or pad
    cout << "Loaded " << N << " samples from " << filename << endl;

    cufftDoubleComplex* d_data;
    cudaMalloc(&d_data, sizeof(cufftDoubleComplex) * N);
    cudaMemcpy(d_data, data.data(), sizeof(cufftDoubleComplex) * N, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_Z2Z, 1);  // Z2Z = double complex

    auto start = chrono::high_resolution_clock::now();

    cufftExecZ2Z(plan, d_data, d_data, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> duration = end - start;
    cout << "cuFFT completed in " << duration.count() << " ms\n";

    cudaMemcpy(data.data(), d_data, sizeof(cufftDoubleComplex) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; ++i) {
        double magnitude = sqrt(data[i].x * data[i].x + data[i].y * data[i].y);
        cout << i << ": " << magnitude << "\n";
    }

    cudaFree(d_data);
    cufftDestroy(plan);

    return 0;
}
