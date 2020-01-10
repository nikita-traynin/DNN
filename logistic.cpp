#include <cmath>

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
	float initial_rate = 1;
	int num_iterations = 500;
	float lower_bound = 0.05;
	if(x > num_iterations)
		return lower_bound;
	else
		return ((lower_bound-initial_rate)/num_iterations)*(x-1) + initial_rate;
}

float Schedule(int x) {
	//return ExponentialSchedule(x);
	return LinearSchedule(x);
}

float RandomWeight(int max_magnitude_times_100) {
	return float((rand()%(2*max_magnitude_times_100+1))-max_magnitude_times_100)/100.0;
}

float Mean(unsigned char *arr, int start, int size) {
	float mean = 0;
	for(int i = 0; i < size; i++) {
		mean += arr[start+i];
	}
	return mean/size;
}

float Variance(unsigned char *arr, int start, int size, float mean) {
	float variance = 0;
	float diff;
	for(int i = 0; i < size; i++) {
		diff = arr[start+i]-mean;
		variance += diff*diff;
	}
	return variance/size;
}

