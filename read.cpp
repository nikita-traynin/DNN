#include <fstream>
#include <iostream>

std::ifstream readTrainingImgHeader(int &magic_num, int &num_img, int &num_row, int &num_col) {
	std::ifstream training_images_file("train-images.idx3-ubyte", std::ios::binary);
	if(!training_images_file.is_open())
		{ std::cout << "\nCouldn't open training image file!\n"; return training_images_file; }
	training_images_file.read((char*)&magic_num, 4);
	training_images_file.read((char*)&num_img, 4);
	training_images_file.read((char*)&num_row, 4);
	training_images_file.read((char*)&num_col, 4);
	return training_images_file;
}

std::ifstream readTestingImgHeader(int &magic_num, int &num_img, int &num_row, int &num_col) {
	std::ifstream testing_images_file("t10k-images.idx3-ubyte", std::ios::binary);
	if(!testing_images_file.is_open())
		{ std::cout << "\nCouldn't open testing image file!\n"; return testing_images_file; }
	testing_images_file.read((char*)&magic_num, 4);
	testing_images_file.read((char*)&num_img, 4);
	testing_images_file.read((char*)&num_row, 4);
	testing_images_file.read((char*)&num_col, 4);
	return testing_images_file;
}

std::ifstream readTrainingLblHeader(int &magic_num, int &num_img) {
	std::ifstream training_labels_file("train-labels.idx1-ubyte", std::ios::binary);
	if(!training_labels_file.is_open())
		{ std::cout << "\nCouldn't open training labels file!\n"; return training_labels_file; }
	training_labels_file.read((char*)&magic_num, 4);
	training_labels_file.read((char*)&num_img, 4);
	return training_labels_file;
}

std::ifstream readTestingLblHeader(int &magic_num, int &num_img) {
	std::ifstream testing_labels_file("t10k-labels.idx1-ubyte", std::ios::binary);
	if(!testing_labels_file.is_open()) 
		{ std::cout << "\nCouldn't open testing labels file!\n"; return testing_labels_file; }
	testing_labels_file.read((char*)&magic_num, 4);
	testing_labels_file.read((char*)&num_img, 4);
	return testing_labels_file;
}