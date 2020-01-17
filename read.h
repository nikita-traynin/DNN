#include <fstream>

std::ifstream readTrainingImgHeader(int &magic_num, int &num_img, int &num_row, int &num_col);
std::ifstream readTestingImgHeader(int &magic_num, int &num_img, int &num_row, int &num_col);
std::ifstream readTrainingLblHeader(int &magic_num, int &num_img);
std::ifstream readTestingLblHeader(int &magic_num, int &num_img);