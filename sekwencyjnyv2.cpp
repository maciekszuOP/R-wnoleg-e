#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <iostream>
#include <complex>
#include <vector>
#include <cmath>
#include <chrono>
#include <stdexcept>
#include <fstream>

const double PI = std::acos(-1);
using Complex = std::complex<double>;
using namespace std;

void fft(vector<Complex>& a) {
    int n = a.size();
    if (n <= 1) return;

    vector<Complex> a_even(n / 2), a_odd(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        a_even[i] = a[i * 2];
        a_odd[i] = a[i * 2 + 1];
    }

    fft(a_even);
    fft(a_odd);

    for (int k = 0; k < n / 2; ++k) {
        Complex t = polar(1.0, -2 * PI * k / n) * a_odd[k];
        a[k] = a_even[k] + t;
        a[k + n / 2] = a_even[k] - t;
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

    // >>> POMIAR FFT – BEZ cout <<<<<<<<<<<<<<
    auto start = chrono::high_resolution_clock::now();
    fft(data);
    auto end = chrono::high_resolution_clock::now();
    // >>> KONIEC POMIARU <<<<<<<<<<<<<<

    chrono::duration<double, milli> duration = end - start;
    cout << "FFT completed in " << duration.count() << " ms\n";

    // >>> cout PO POMIARZE
    for (int i = 0; i < 10; ++i) {
        cout << i << ": " << abs(data[i]) << "\n";
    }

    return 0;
}
