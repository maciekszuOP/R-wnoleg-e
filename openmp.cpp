#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <fstream>
#include <omp.h>

const double PI = std::acos(-1);
using Complex = std::complex<double>;
using namespace std;

void fft_openmp(vector<Complex>& a) {
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

    for (int s = 1; s <= log_n; ++s) {
        int m = 1 << s;
        Complex wm = polar(1.0, -2 * PI / m);

        #pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k += m) {
            Complex w = 1;
            for (int j = 0; j < m / 2; ++j) {
                Complex t = w * a[k + j + m / 2];
                Complex u = a[k + j];
                a[k + j] = u + t;
                a[k + j + m / 2] = u - t;
                w *= wm;
            }
        }
    }
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

    auto start = chrono::high_resolution_clock::now();
    fft_openmp(data);
    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double, milli> duration = end - start;
    cout << "FFT completed in " << duration.count() << " ms\n";

    for (int i = 0; i < 10; ++i) {
        cout << i << ": " << abs(data[i]) << "\n";
    }

    return 0;
}
//g++ -fopenmp -O2 openmp.cpp -o openmp.exe  