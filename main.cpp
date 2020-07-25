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
const int kNodeCount[kLayerCount-2] {128, 64};		//>=1 (for hidden layers only)			
const int kBatchSize = 20;						//TODO: currently must be 1! Change so that different batchsizes are possible.
const int kHighErrorCount = 10;					//how many most difficult test images to show
const int kEpochCount = 8;

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
	
	//PrintNumbers(training_pixels, kResolution, 10, 15);
	
	/****************DECLARE NETWORK*****************/
	int num_nodes[kLayerCount];
	num_nodes[0] = kInputCount;
	for(int i = 1; i < kLayerCount-1; i++) {
		num_nodes[i] = kNodeCount[i-1];
	}
	num_nodes[kLayerCount-1] = 10;					
	Layer layers[kLayerCount];
	/************************************************/
	
	/****************INITIALIZE WEIGHTS*****************/
	layers[0].a_.assign(num_nodes[0],0);							//input layer doesn't get weights or biases
	for(int i = 1; i < kLayerCount; i++) {
		//0- initialize activations
		layers[i].a_.assign(num_nodes[i],0);
		//boundary of uniform distribution for weights/biases [-max_magnitude, max_magnitude]
		float max_magnitude = 1.0 / sqrt(num_nodes[i-1]);
		for(int k = 0; k < num_nodes[i]; k++) {
			layers[i].b_.push_back(SymmetricUniform(max_magnitude));
		}
		for(int j = 0; j < num_nodes[i-1]; j++) {
			vector<float> temp;
			for(int k = 0; k < num_nodes[i]; k++) {
				temp.push_back(SymmetricUniform(max_magnitude));
			}
			layers[i].w_.push_back(temp);								//TO the ith layer
		}
	}
	/***************************************************/
	
	/****************ALLOCATE GRADIENTS*****************/
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
	/****************************************************/
	
	
	int num_testing_errors = 0;
	float learning_rate = 0;
	//Training Loop
	for(int epoch_iter = 0; epoch_iter < kEpochCount; epoch_iter++) {
		for(int batch_iter = 0; batch_iter < kTrainingImageCount/kBatchSize; batch_iter++) {
			
			/****************CLEAR GRADIENTS*****************/
			for(int i = 0; i < kLayerCount; i++) {
				for(int j = 0; j < num_nodes[i]; j++) {
					if(i != 0) {
						bias_gradient[i-1][j] = 0;
					}
					for(int k = 0; k < num_nodes[i+1]; k++) {
						if(i != kLayerCount-1) {
							weight_gradient[i][j][k] = 0;
						}
					}
				}
			}
			/************************************************/
			
			//iterating through images in batch
			for(int iter = 0; iter < kBatchSize; iter++) {
				int image_index = batch_iter*kBatchSize + iter;
				learning_rate = 0.001;
				
				/****************FEEDFORWARD*****************/
				for(int i = 0; i < num_nodes[0]; i++) {
					layers[0].a_[i] = training_pixels[image_index*kInputCount + i ] / 256.0;
				}
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
				/********************************************/
				
				/* if(image_index < 5) {
					cout << "\n";
					cout << int(training_labels[image_index]) << ":\n";
					for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
						cout << layers[kLayerCount-1].a_[i] << "\n";
					}
				} */

				/****************BACKPROPOGATE*****************/
				for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
					if(i == training_labels[image_index]) {
						activation_gradient[kLayerCount-1][i] = (layers[kLayerCount-1].a_[i]-1);
					}
					else {
						activation_gradient[kLayerCount-1][i] = (layers[kLayerCount-1].a_[i]);
					}
				}
				for(int i = kLayerCount-2; i >= 0; i--) {
					for(int j = 0; j < num_nodes[i]; j++) {
						float sum = 0;
						for(int k = 0; k < num_nodes[i+1]; k++) {
							//IMPORTANT: weight gradient calculation 
							weight_gradient[i][j][k] += activation_gradient[i+1][k]*ReLUPrime(layers[i+1].a_[k])*layers[i].a_[j];
							//Matrix multiply to get activation gradient (W_j * X_j * X_j')
							if(i != 0)
								sum += layers[i+1].w_[j][k]*activation_gradient[i+1][k]*ReLUPrime(layers[i+1].a_[k]);
						}
						activation_gradient[i][j] = sum;
					}
					for(int j = 0; j < num_nodes[i+1]; j++) {
						bias_gradient[i][j] += ReLUPrime(layers[i+1].a_[j])*activation_gradient[i+1][j];
					}  
				}
				/**********************************************/
				
				//Display progress
				if(image_index % 1000 == 0) {
					cout << "\n" << image_index << " images complete. ";
				}
			}
			
			/****************ADJUST WEIGHTS AND BIASES*****************/
			for(int i = 0; i < kLayerCount; i++) {
				for(int j = 0; j < num_nodes[i]; j++) {
					if(i != kLayerCount-1) {
						for(int k = 0; k < num_nodes[i+1]; k++) {
							layers[i+1].w_[j][k] -= learning_rate*weight_gradient[i][j][k]/kBatchSize;
						}
					}
					if(i != 0) {
						layers[i].b_[j] -= learning_rate*bias_gradient[i-1][j]/kBatchSize;
					}
				}
			}
			/**********************************************************/
		}
		
		//GET VALIDATION ERROR
		num_testing_errors = 0;
		for(int iter = 0; iter < kTestingImageCount; iter++) {
			
			/****************FEEDFORWARD*****************/
			for(int i = 0; i < num_nodes[0]; i++) {
				layers[0].a_[i] = testing_pixels[iter*kInputCount + i ] / 256.0;
			}
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
			/********************************************/
			
			/****************CALCULATE ERROR*****************/
			float max = -1;
			int predicted_label = -1;											
			for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
				float activation = layers[kLayerCount-1].a_[i];
				if(activation > max) { 
					predicted_label = i; 
					max = activation; 
				}
			}
			if(predicted_label != int(testing_labels[iter])) {
				num_testing_errors++;
			}
			/************************************************/
		}
		
		//Display Validation Error
		cout << "\nEpoch " << epoch_iter+1 << ": Validation error rate is: " << setw(3) << float(num_testing_errors)/kTestingImageCount * 100.0 << "%.\n";
	}
	
	
	
	//TESTING LOOP//
	float cost, min_max_cost = 0;
	num_testing_errors = 0;
	vector<float> high_error_stats;
	for(int iter = 0; iter < kTestingImageCount; iter++) {
		
		/****************FEEDFORWARD*****************/
		for(int i = 0; i < num_nodes[0]; i++) {
			layers[0].a_[i] = testing_pixels[iter*kInputCount + i ] / 256.0;
		}
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
		/********************************************/
		
		/****************CALCULATE ERROR*****************/
		float max = -1;
		int predicted_label = -1;						
		cost = 0;
		for(int i = 0; i < num_nodes[kLayerCount-1]; i++) {
			float activation = layers[kLayerCount-1].a_[i];
			if(i == (int)testing_labels[iter]) {
				cost += 0.5*(activation-1)*(activation-1);
			}
			else {
				cost += 0.5*activation*activation;
			}
			if(activation > max)
				{ predicted_label = i; max = activation; }
		}
		/************************************************/

		//Add to error stats.
		if(predicted_label != (int)testing_labels[iter]) {
			num_testing_errors++;
			if(num_testing_errors <= kHighErrorCount) {
				high_error_stats.push_back(float(testing_labels[iter]));
				high_error_stats.push_back(float(predicted_label));
				high_error_stats.push_back(cost);
				high_error_stats.push_back(max);
				high_error_stats.push_back(float(iter+1));
				min_max_cost = high_error_stats[MinimumCostIndex(high_error_stats)];
				
			}
			else if (cost > min_max_cost) {
				float index = MinimumCostIndex(high_error_stats);
				high_error_stats[index-2] = float(testing_labels[iter]);
				high_error_stats[index-1] = float(predicted_label);
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
	cout << "\nThe testing error is: " << float(num_testing_errors)/kTestingImageCount*100.0 << "%.\n";
	cout << "\nThe end!\n";
}
