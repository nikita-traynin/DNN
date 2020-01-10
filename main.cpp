//Welcome to my neural networks program! (new version, NOT DONE)
//This is about setting up and training a network to predict grayscale hand-written digits
//We start with a resolution of 28-by-28, using the MNIST database. http://yann.lecun.com/exdb/mnist/
//
//By: Nikita Traynin
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <ctime>
#include "logistic.h"						//includes cmath
#include "layer.h"							//includes vector

using namespace std;

//hyperparameters
const int kLayerCount = 4;						//>=3 (one for input, one for decision, and at least one hidden)					
const int kNodeCount[kLayerCount-2] {50,20};	//>=1 			
const int kBatchSize = 1;						//TODO: currently must be 1! Change so that different batchsizes are possible.
const int kInitialWeightMax = 2;				//max magnitude of initial random weights times 100

//TODO temporary variables for input-data related info. Remove when we can read this from input file (idx1/idx3)
const int kResolution = 28; 									
const int kInputCount = kResolution*kResolution;								
const int kTrainingImageCount = 60000;			
const int kTestingImageCount = 10000;					

int main() {
	//set random seed
	srand(time(NULL));
	
	//pixels/training_labels dynamic due to large array size
	unsigned char* training_pixels = new unsigned char[kTrainingImageCount*kInputCount];  
	unsigned char* training_labels = new unsigned char[kTrainingImageCount];
	unsigned char* testing_pixels = new unsigned char[kTestingImageCount*kInputCount];
	unsigned char* testing_labels = new unsigned char[kTestingImageCount];
	int magic_num, num_img, num_row, num_col, num_items;									//variables to store data from the headers of NMIST files
	
	//Read header of training images file
	ifstream training_images_file("train-images.idx3-ubyte", ios::binary);
	if(!training_images_file.is_open())
		{ cout << "\nCouldn't open training image file!\n"; return -1; }
	training_images_file.read((char*)&magic_num, 4);
	training_images_file.read((char*)&num_img, 4);
	training_images_file.read((char*)&num_row, 4);
	training_images_file.read((char*)&num_col, 4);
	
	//Read header of training images file
	ifstream testing_images_file("t10k-images.idx3-ubyte", ios::binary);
	if(!testing_images_file.is_open())
		{ cout << "\nCouldn't open testing image file!\n"; return -1; }
	testing_images_file.read((char*)&magic_num, 4);
	testing_images_file.read((char*)&num_img, 4);
	testing_images_file.read((char*)&num_row, 4);
	testing_images_file.read((char*)&num_col, 4);
	
	//Read header of training training_labels file
	ifstream training_labels_file("train-labels.idx1-ubyte", ios::binary);
	if(!training_labels_file.is_open())
		{ cout << "\nCouldn't open training label file!\n"; return -1; }
	training_labels_file.read((char*)&magic_num, 4);
	training_labels_file.read((char*)&num_items, 4);
	
	//Read header of training training_labels file
	ifstream testing_labels_file("t10k-labels.idx1-ubyte", ios::binary);
	if(!testing_labels_file.is_open())
		{ cout << "\nCouldn't open testing label file!\n"; return -1; }
	testing_labels_file.read((char*)&magic_num, 4);
	testing_labels_file.read((char*)&num_items, 4);

	//Read pixel/label training data (slow!)
	for(int i = 0; i < kTrainingImageCount; i++) {
		for(int j = 0; j < kInputCount; j++) {
			training_images_file.read((char*)(training_pixels + i*kInputCount+j), 1);
		}
		training_labels_file.read((char*)(training_labels+i) ,1);
	}
	
	//Read pixel/label testing data (slow!)
	for(int i = 0; i < kTestingImageCount; i++) {
		for(int j = 0; j < kInputCount; j++) {
			testing_images_file.read((char*)(testing_pixels+i*kInputCount+j), 1);
		}
		testing_labels_file.read((char*)(testing_labels+i) ,1);
	}
	
	//Initialize network objects and set the number of nodes in each layer
	int num_nodes[kLayerCount];
	num_nodes[0] = kInputCount;
	for(int i = 1; i < kLayerCount-1; i++) {
		num_nodes[i] = kNodeCount[i-1];
	}
	num_nodes[kLayerCount-1] = 10;					
	Layer layers[kLayerCount];
	
	//initialize default weights and activations for the network
	layers[0].a_.assign(num_nodes[0],0);							//input layer doesn't get weights or biases
	for(int i = 1; i < kLayerCount; i++) {
		layers[i].a_.assign(num_nodes[i],0);
		for(int k = 0; k < num_nodes[i]; k++) {
			layers[i].b_.push_back(RandomWeight(100));
		}
		for(int j = 0; j < num_nodes[i-1]; j++) {
			vector<float> temp;
			for(int k = 0; k < num_nodes[i]; k++) {
				temp.push_back(RandomWeight(kInitialWeightMax*100));
			}
			layers[i].w_.push_back(temp);								//TO the ith layer
		}
	}
	
	
	//Initialize variables and arrays for training
	int num_training_errors = 0;
	float **activation_gradient = new float*[kLayerCount];
	activation_gradient[kLayerCount-1] = new float[num_nodes[kLayerCount-1]];
	float ***weight_gradient = new float**[kLayerCount-1];					//FROM the ith layer
	float **bias_gradient = new float*[kLayerCount-1];
	
	//TRAINING LOOP (slow)//
	for(int iter = 0; iter < kTrainingImageCount/kBatchSize; iter++) {
		float learning_rate = Schedule(iter+1);
		float mean = Mean(training_pixels, iter*num_nodes[0], num_nodes[0]);
		float variance = Variance(training_pixels, iter*num_nodes[0], num_nodes[0], mean);
		
		//Plug in normalized (mean 0 variance 1) pixel values
		for(int i = 0; i < num_nodes[0]; i++) {
			layers[0].a_[i] = (1/sqrt(variance))*(training_pixels[iter*num_nodes[0] + i]-mean);				
		}

		//FEEDFORWARD//
		for(int i = 1; i < kLayerCount; i++) {
			for(int j = 0; j < num_nodes[i]; j++) {
				float sum = 0;
				for(int k = 0; k < num_nodes[i-1]; k++) {
					sum += layers[i-1].a_[k] * layers[i].w_[k][j];
				}
				sum += layers[i].b_[j];						
				layers[i].a_[j] = Logistic(sum);			
			}
		}
		
		//Count the errors
		int max_index = -1;
		float max = -1;
		for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
			if(layers[kLayerCount-1].a_[i] > max) {
				max_index = i; 
				max = layers[kLayerCount-1].a_[i];
			}
		}
		if(max_index != training_labels[iter])
			num_training_errors++;
		
		
		//Calculate cost and final layer gradient
		float cost = 0;
		float diff;
		for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
			if(i == training_labels[iter]) {
				activation_gradient[kLayerCount-1][i] = 2*(layers[kLayerCount-1].a_[i]-1);
				diff = layers[kLayerCount-1].a_[i]-1;
			}
			else {
				activation_gradient[kLayerCount-1][i] = 2*(layers[kLayerCount-1].a_[i]);
				diff = layers[kLayerCount-1].a_[i];
			}
			cost += diff*diff;
		}
		
		//Backpropagate to find gradients for all other layers
		for(int i = kLayerCount-2; i >= 0; i--) {
			activation_gradient[i] = new float[num_nodes[i]];
			weight_gradient[i] = new float*[num_nodes[i]];
			bias_gradient[i] = new float[num_nodes[i+1]];
			for(int j = 0; j < num_nodes[i]; j++) {
				weight_gradient[i][j] = new float[num_nodes[i+1]];
				float sum = 0;
				for(int k = 0; k < num_nodes[i+1]; k++) {
					float factor = LogisticPrime(layers[i+1].a_[k]);
					weight_gradient[i][j][k] = layers[i].a_[j]*factor*activation_gradient[i+1][k];
					if(i != 0)
						sum += layers[i+1].w_[j][k]*factor*activation_gradient[i+1][k];
				}
				activation_gradient[i][j] = sum;
			}
			for(int j = 0; j < num_nodes[i+1]; j++) {
				bias_gradient[i][j] = LogisticPrime(layers[i+1].a_[j])*activation_gradient[i+1][j];
			}
			delete [] activation_gradient[i];
		}
		
		//Adjust weights and biases
		for(int i = 0; i < kLayerCount; i++) {
			for(int j = 0; j < num_nodes[i]; j++) {
				if(i != kLayerCount-1) {
					for(int k = 0; k < num_nodes[i+1]; k++) {
						layers[i+1].w_[j][k] -= learning_rate*weight_gradient[i][j][k];
					}
					delete [] weight_gradient[i][j];
				}
				if(i != 0) {
					layers[i].b_[j] -= learning_rate*bias_gradient[i-1][j];
				}
			}
			delete [] weight_gradient[i];
			if(i != 0)
				delete [] bias_gradient[i-1];
		}
		
		//TODO: temporary, only for SGD, checking every 500 iterations various variables.
		if(iter%500 == 0) {
			cout << "\nImage " << iter+1;
			cout << ". Cost: " << cost << ". Learning rate: " << learning_rate << ". So far error: " << float(num_training_errors)/float(iter+1)*100.0 << "%. So far missed: " << num_training_errors;
		}
	}
	
	//TESTING LOOP//
	int num_testing_errors = 0;
	for(int iter = 0; iter < kTestingImageCount; iter++) {
		float mean = Mean(testing_pixels, iter*num_nodes[0], num_nodes[0]);
		float variance = Variance(testing_pixels, iter*num_nodes[0], num_nodes[0], mean);
		
		//Plug in normalized pixel values
		for(int i = 0; i < num_nodes[0]; i++) {
			layers[0].a_[i] = (1/sqrt(variance))*(testing_pixels[iter*num_nodes[0] + i]-mean);					//mean-0, variance-1
		}
		
		//Feed into rest of network
		for(int i = 1; i < kLayerCount; i++) {
			for(int j = 0; j < num_nodes[i]; j++) {
				float sum = 0;
				for(int k = 0; k < num_nodes[i-1]; k++) {
					sum += layers[i-1].a_[k] * layers[i].w_[k][j];
				}
				sum += layers[i].b_[j];				
				layers[i].a_[j] = Logistic(sum);			
			}
		}
		
		//Find which number the networks reads input as
		float max = 0;
		int max_index = -1;
		for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
			if(layers[kLayerCount-1].a_[i] > max)
				{ max_index = i; max = layers[kLayerCount-1].a_[i]; }
		}

		//If it's the wrong number, indicate an error
		if(max_index != (int)testing_labels[iter]) {
			num_testing_errors++;
			if(num_testing_errors < 25)
				cout << "\nMissed guess on " << iter+1 << ": " << max_index << ". Actual: " << (int)testing_labels[iter] << ". Highest activation: " << max;
		}
	}
	cout << "\nThe testing error is: " << (float(num_testing_errors)/kTestingImageCount)*100 << "%.\n";
	cout << "\nThe end!\n";
}
