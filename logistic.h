float Logistic(float x);
float LogisticPrime(float x);
float ReLU(float x);
float Schedule(int x);
float ExponentialSchedule(int x);
float LinearSchedule(int x);
float RandomWeight(int max_magnitude_times_100);
float Mean(unsigned char *arr, int start, int size);
float Variance(unsigned char *arr, int start, int size, float mean);