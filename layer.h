#include <vector>

using namespace std;

struct Layer {
	vector<float> a_;					//node activations
	vector< vector<float> > w_;			//weights of connections to this layer
	vector<float> b_;					//node biases
};
