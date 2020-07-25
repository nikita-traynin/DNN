#include <cmath>
#include <vector>
#include <iostream>

using namespace std;

float Logistic(float x) {
	float k = 1/1.5;
	return 1/(1+exp(-k*x));
}

float LogisticPrime(float x) {
	float k = 1/1.5;
	return x*(1-x)*k;
}

float ReLU(float x) {
	if (x<0)
		return 0;
	else 
		return x;
}

float ReLUPrime(float ReLUresult) {
	if (ReLUresult > 0) {
		return 1;
	}
	else
		return 0;
}

float ExponentialSchedule(int x) {
	float multiplier = 5;
	float decay_scaling = 29;													//the higher, the slower learning rate decays
	float lower_bound = 0.05;
	if((x-1) > decay_scaling*log(multiplier/lower_bound))
		return lower_bound;
	else
		return multiplier*exp(-(x-1)/decay_scaling);
}

float LinearSchedule(int x) {
	float initial_rate = 0.1;
	int num_iterations = 10;
	float lower_bound = 0.1;
	if(x > num_iterations)
		return lower_bound;
	else
		return ((lower_bound-initial_rate)/num_iterations)*(x-1) + initial_rate;
}

float Schedule(int x) {
	//return ExponentialSchedule(x);
	return LinearSchedule(x);
}

float SymmetricUniform(float max_magnitude) {
	return (rand()%(int(2*max_magnitude*100.0+1)) - int(max_magnitude*100.0))/100.0;
}

int MinimumCostIndex(vector<float> &vec) {
	float minimum = 10;
	int index = -1;
	for(int i = 0; i < vec.size()/5; i++) {
		if (vec[i*5+2] < minimum) {
			index = i*5+2;
			minimum = vec[index];
		}
	}
	return index;
}

void PrintNumbers(unsigned char* pixels, int resolution, int start, int end) {
	for(int image_count = start; image_count < end; image_count++) {
		cout << "\n\n";
		for(int i = 0; i < resolution; i++) {
			for(int j = 0; j < resolution; j++) {
				int pixel = int(*(pixels + image_count*resolution*resolution + i*resolution + j));
				if (pixel > 0) {
					cout << " 1 ";
				}
				else {
					cout << "000";
				}
			}
			cout << "\n";
		}
	}
	return;
}
