#include <vector>

// This file contains functions related to activation function, schedule/learning rate, 
// functions needed for normalizing input, generating random weights, and finding min(vector).


float Logistic(float x);
float LogisticPrime(float x);
float ReLU(float x);
float ReLUPrime(float ReLUresult);
float Schedule(int x);
float ExponentialSchedule(int x);
float LinearSchedule(int x);
float SymmetricUniform(float max_magnitude);
int MinimumCostIndex(std::vector<float> &vec);
void PrintNumbers(unsigned char* pixels, int resolution, int start, int end);