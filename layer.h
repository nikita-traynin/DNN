#include <vector>

struct Layer {
	std::vector<float> a_;					//node activations
	std::vector< std::vector<float> > w_;			//weights of connections to this layer
	std::vector<float> b_;					//node biases
};

