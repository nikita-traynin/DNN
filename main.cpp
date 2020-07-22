//Welcome to my neural networks program! (new version, NOT DONE)
//This is about setting up and training a network to predict hand-written digits
//We start with a resolution of 28-by-28, using the MNIST database. http://yann.lecun.com/exdb/mnist/
//By: Nikita Traynin

#include <iostream>		//cout 
#include <fstream>		//ifstream
#include <iomanip>		//setw	
#include <cmath>		//sqrt
#include <ctime>		//time(NULL) (for rand)
#include <vector>		//for vectors
#include "logistic.h"						
#include "layer.h"							
#include "read.h"	

using namespace std;

//hyperparameters
const int kLayerCount = 4;						//>=3 (one for input, one for decision, and at least one hidden)					
const int kNodeCount[kLayerCount-2] {50, 25};		//>=1 			
const int kBatchSize = 1;						//TODO: currently must be 1! Change so that different batchsizes are possible.
const int kHighErrorCount = 10;					//how many most difficult test images to show
const int kEpochCount = 3;

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
	
	ifstream training_images_file = readTrainingImgHeader(magic_num, num_img, num_row, num_col);
	ifstream testing_images_file = readTestingImgHeader(magic_num, num_img, num_row, num_col);
	ifstream training_labels_file = readTrainingLblHeader(magic_num, num_img);
	ifstream testing_labels_file = readTestingLblHeader(magic_num, num_img);

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
	
	
	//Print out a number from training set
	for(int number_i = 0; number_i < 5; number_i++) {
		cout << "\n\n\n";
		for(int i  = 0; i < kResolution; i++) {
			for(int j = 0; j < kResolution; j++) {
				int pixel = int(*(training_pixels + number_i*kInputCount + i*kResolution + j));
				if (pixel > 0) {
					cout << " 1 ";
				}
				else
					cout << "000"; 
			}
			cout << "\n";
		}
		cout << "\n\n";
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
		float max_magnitude = 1.0 / sqrt(num_nodes[i]);
		layers[i].a_.assign(num_nodes[i],0);
		for(int k = 0; k < num_nodes[i]; k++) {
			layers[i].b_.push_back(0);
		}
		for(int j = 0; j < num_nodes[i-1]; j++) {
			vector<float> temp;
			for(int k = 0; k < num_nodes[i]; k++) {
				temp.push_back(RandomWeight(max_magnitude));
			}
			layers[i].w_.push_back(temp);								//TO the ith layer
		}
	}
	
	//Initialize variables and arrays for training
	float **activation_gradient = new float*[kLayerCount];
	float ***weight_gradient = new float**[kLayerCount-1];					//FROM the ith layer
	float **bias_gradient = new float*[kLayerCount-1];
	for(int i = 0; i < kLayerCount; i++) {
		if(i != kLayerCount-1) {
			weight_gradient[i] = new float*[num_nodes[i]];
			bias_gradient[i] = new float[num_nodes[i+1]];
			for(int j = 0; j < num_nodes[i]; j++) {
				weight_gradient[i][j] = new float[num_nodes[i+1]];
			}
		}
		activation_gradient[i] = new float[num_nodes[i]];
	}
	
	//TRAINING LOOP (slow)//
	int num_training_errors = 0, max_index = -1;
	float cost = 0, max = -1, activation = -1, min_max_cost = 0;
	for(int epoch_iter = 0; epoch_iter < kEpochCount; epoch_iter++) {
		for(int iter = 0; iter < kTrainingImageCount/kBatchSize; iter++) {
			float learning_rate = Schedule(epoch_iter*kTrainingImageCount + iter+1);
			
			//Plug in normalized ( range (0,1) ) pixel values
			for(int i = 0; i < num_nodes[0]; i++) {
				layers[0].a_[i] = training_pixels[iter*num_nodes[0] + i ] / 256.0;
			}

			//FEEDFORWARD//
			for(int i = 1; i < kLayerCount; i++) {
				for(int j = 0; j < num_nodes[i]; j++) {
					float sum = 0;
					for(int k = 0; k < num_nodes[i-1]; k++) {
						sum += layers[i-1].a_[k] * layers[i].w_[k][j];
					}
					sum += layers[i].b_[j];						
					layers[i].a_[j] = ReLU(sum);			
				}
			}
			
			//Calculate cost and final layer gradient
			float cost = 0;
			float diff;
			for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
				if(i == training_labels[iter]) {
					activation_gradient[kLayerCount-1][i] = (layers[kLayerCount-1].a_[i]-1);
					diff = layers[kLayerCount-1].a_[i]-1;
				}
				else {
					activation_gradient[kLayerCount-1][i] = (layers[kLayerCount-1].a_[i]);
					diff = layers[kLayerCount-1].a_[i];
				}
				cost += diff*diff;
			}
			
			//Backpropagate to find gradients for all other layers
			for(int i = kLayerCount-2; i >= 0; i--) {
				for(int j = 0; j < num_nodes[i]; j++) {
					float sum = 0;
					for(int k = 0; k < num_nodes[i+1]; k++) {
						//IMPORTANT: weight gradient calculation 
						weight_gradient[i][j][k] = activation_gradient[i+1][k]*ReLUPrime(layers[i+1].a_[k])*layers[i].a_[j];
						//Matrix multiply to get activation gradient (W_j * X_j * X_j')
						if(i != 0)
							sum += layers[i+1].w_[j][k]*activation_gradient[i+1][k]*ReLUPrime(layers[i+1].a_[k]);
					}
					activation_gradient[i][j] = sum;
				}
				for(int j = 0; j < num_nodes[i+1]; j++) {
					bias_gradient[i][j] = ReLUPrime(layers[i+1].a_[j])*activation_gradient[i+1][j];
				}  
			}
			
			//Adjust weights and biases
			for(int i = 0; i < kLayerCount; i++) {
				for(int j = 0; j < num_nodes[i]; j++) {
					if(i != kLayerCount-1) {
						for(int k = 0; k < num_nodes[i+1]; k++) {
							layers[i+1].w_[j][k] -= learning_rate*weight_gradient[i][j][k];
						}
						//delete [] weight_gradient[i][j];
					}
					if(i != 0) {
						layers[i].b_[j] -= learning_rate*bias_gradient[i-1][j];
					}
				}
				//delete [] weight_gradient[i];
				// if(i != 0)
					// delete [] bias_gradient[i-1];
			}
			if(iter % 1500 == 0) {
				cout << "\n" << iter << " images complete. ";
			}
		}
		//GET TRAINING ERROR
		num_training_errors = 0;
		for(int iter = 0; iter < kTrainingImageCount; iter++) {
			
			//Plug in normalized ( range (0,1) ) pixel values
			for(int i = 0; i < num_nodes[0]; i++) {
				layers[0].a_[i] = training_pixels[iter*num_nodes[0] + i ] / 256.0;
			}
			
			//Feed into rest of network
			for(int i = 1; i < kLayerCount; i++) {
				for(int j = 0; j < num_nodes[i]; j++) {
					float sum = 0;
					for(int k = 0; k < num_nodes[i-1]; k++) {
						sum += layers[i-1].a_[k] * layers[i].w_[k][j];
					}
					sum += layers[i].b_[j];				
					layers[i].a_[j] = ReLU(sum);			
				}
			}
			
			//Count errors
			max = -1;
			max_index = -1;											//Max index is the digit the network predicted (max because it's the maximum activation)
			for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
				activation = layers[kLayerCount-1].a_[i];
				if(activation > max)
					{ max_index = i; max = activation; }
			}
			if(max_index != training_labels[iter]) {
				num_training_errors++;
			}
		}
		
		float training_percent_error = float(num_training_errors)/kTrainingImageCount * 100.0;
		cout << "\n\nValidation error rate is: " << setw(3) << training_percent_error << "%.\n";
	}
	
	
	
	//TESTING LOOP//
	int num_testing_errors = 0;
	vector<float> high_error_stats;
	for(int iter = 0; iter < kTestingImageCount; iter++) {
		
		//Plug in normalized ( range (0,1) ) pixel values
		for(int i = 0; i < num_nodes[0]; i++) {
			layers[0].a_[i] = training_pixels[iter*num_nodes[0] + i ] / 256.0;
		}
		
		//Feed into rest of network
		for(int i = 1; i < kLayerCount; i++) {
			for(int j = 0; j < num_nodes[i]; j++) {
				float sum = 0;
				for(int k = 0; k < num_nodes[i-1]; k++) {
					sum += layers[i-1].a_[k] * layers[i].w_[k][j];
				}
				sum += layers[i].b_[j];				
				layers[i].a_[j] = ReLU(sum);			
			}
		}
		
		//Find which number the networks reads input as
		max = -1;
		max_index = -1;											//Max index is the digit the network predicted (max because it's the maximum activation)
		cost = 0;
		activation = -1;
		for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
			activation = layers[kLayerCount-1].a_[i];
			if(i == (int)testing_labels[iter]) {
				cost += (activation-1)*(activation-1);
			}
			else {
				cost += activation*activation;
			}
			if(activation > max)
				{ max_index = i; max = activation; }
		}

		//If it's the wrong number, indicate an error
		if(max_index != (int)testing_labels[iter]) {
			num_testing_errors++;
			if(num_testing_errors <= kHighErrorCount) {
				high_error_stats.push_back(float(testing_labels[iter]));
				high_error_stats.push_back(float(max_index));
				high_error_stats.push_back(cost);
				high_error_stats.push_back(max);
				high_error_stats.push_back(float(iter+1));
				min_max_cost = high_error_stats[MinimumCostIndex(high_error_stats)];
				
			}
			else if (cost > min_max_cost) {
				float index = MinimumCostIndex(high_error_stats);
				high_error_stats[index-2] = float(testing_labels[iter]);
				high_error_stats[index-1] = float(max_index);
				high_error_stats[index] = cost;
				high_error_stats[index+1] = max;
				high_error_stats[index+2] = float(iter+1);
				min_max_cost = high_error_stats[index];
			}
		}
	}
	cout << "\n" << kHighErrorCount << " most difficult digits: ";
	for(int i = 0; i < kHighErrorCount; i++) {
		cout << "\nAt image " << setw(6) << high_error_stats[i*5+4] << ": Actual " << high_error_stats[i*5] << ", Predicted " << high_error_stats[i*5+1] << ", Cost " << setw(4) << high_error_stats[i*5+2] << ", Max Activation " << setw(4) << high_error_stats[i*5+3];
	}
	cout << "\nThe testing error is: " << (float(num_testing_errors)/kTestingImageCount)*100 << "%.\n";
	cout << "\nThe end!\n";
}
