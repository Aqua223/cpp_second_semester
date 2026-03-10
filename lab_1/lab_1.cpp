#include <iostream>
#include <vector>
#include <random>
#include <cmath>

using namespace std;

class IDataSample {
public:
	virtual ~IDataSample() = default;
	virtual vector<float> getFeatures() const = 0;
	virtual vector<float> getTarget() const = 0;
	virtual size_t getFeatureSize() const = 0;
	virtual size_t getTargetSize() const = 0;
};

class NumericDataSample : public IDataSample {
private:
	vector<float> features;
	vector<float> target;

public:
	NumericDataSample(const vector<float>& f, const vector<float>& t)
		: features(f), target(t) {
	}

	vector<float> getFeatures() const override { return features; }
	vector<float> getTarget() const override { return target; }
	size_t getFeatureSize() const override { return features.size(); }
	size_t getTargetSize() const override { return target.size(); }
};

class LearningModule {
private:
	int input_size;
	int hidden_size;
	int output_size;

	vector<vector<float>> W1, W2;
	vector<float> b1, b2;

	vector<unique_ptr<IDataSample>> dataset;

public:
	LearningModule(int input_size, int hidden_size, int output_size)
		: input_size(input_size),
		  hidden_size(hidden_size),
		  output_size(output_size) {

		W1.resize(hidden_size, vector<float>(input_size));
		W2.resize(output_size, vector<float>(hidden_size));

		b1.resize(hidden_size);
		b2.resize(output_size);

		initialize_weights();

		cout << "Module created\n";
		cout << "Architecture: " << input_size << ' ' << hidden_size << ' ' << output_size << '\n';
	}

	void add_data_sample(unique_ptr<IDataSample> sample) {
		if (!dataset.empty()) {
			if (sample->getFeatureSize() != input_size) {
				throw runtime_error("Incompatible feature size!");
			}
			if (sample->getTargetSize() != output_size) {
				throw runtime_error("Incompatible target size!");
			}
		}
		dataset.push_back(move(sample));
	}

	void forward(const vector<float>& input, 
		vector<float>& hidden,
		vector<float>& output) {

		if (input.size() != input_size) {
			throw runtime_error("Incompatible input layer size!");
		}

		for (int i = 0; i < hidden_size; i++) {
			float sum = b1[i]; 
			for (int j = 0; j < input_size; j++) {
				sum += W1[i][j] * input[j];
			}
			hidden[i] = sigmoid(sum); 
		}

		for (int i = 0; i < output_size; i++) {
			float sum = b2[i];
			for (int j = 0; j < hidden_size; j++) {
				sum += W2[i][j] * hidden[j];
			}
			output[i] = sigmoid(sum);
		}
	}

	void backward(const vector<float>& input,
		const vector<float>& target,
		const vector<float>& hidden,
		const vector<float>& output) {

		std::vector<float> grad_output(output_size);
		for (int i = 0; i < output_size; i++) {
	
			float diff = output[i] - target[i];
			grad_output[i] = 2 * diff * output[i] * (1 - output[i]);
		}

		vector<float> grad_hidden(hidden_size, 0.0f);
		for (int i = 0; i < hidden_size; i++) {
			float sum = 0.0f;
			for (int j = 0; j < output_size; j++) {
				sum += W2[j][i] * grad_output[j];
			}
			grad_hidden[i] = sum * hidden[i] * (1 - hidden[i]);
		}


		for (int i = 0; i < output_size; i++) {
			for (int j = 0; j < hidden_size; j++) {
				W2[i][j] -= learning_rate * grad_output[i] * hidden[j];
			}
			b2[i] -= learning_rate * grad_output[i];
		}


		for (int i = 0; i < hidden_size; i++) {
			for (int j = 0; j < input_size; j++) {
				W1[i][j] -= learning_rate * grad_hidden[i] * input[j];
			}
			b1[i] -= learning_rate * grad_hidden[i];
		}
	}

	float train_step(const vector<float>& input,
		const vector<float>& target) {

		auto hidden = vector<float>(hidden_size);
		auto output = vector<float>(output_size);

		forward(input, hidden, output);

		float loss = mseLoss(output, target);

		backward(input, target, hidden, output);

		return loss;
	}

	void set_learning_rate(float lr) {
		learning_rate = lr;
	}

	float train_epoch() {
		float totalLoss = 0.0f;

		for (const auto& sample : dataset) {
			auto features = sample->getFeatures();
			auto target = sample->getTarget();

			float loss = train_step(features, target);
			totalLoss += loss;
		}

		return totalLoss / dataset.size();
	}

	void train(int epochs, bool verbose = true) {
		if (dataset.empty()) {
			cout << "No data for learning!\n";
			return;
		}

		for (int epoch = 0; epoch < epochs; epoch++) {
			float avgLoss = train_epoch();

			if (verbose) {
				cout << "Epoch " << epoch + 1 << "/" << epochs
					<< ", mean loss: " << avgLoss << "\n";
			}
		}
	}

private:
	void initialize_weights() {
		random_device rd;
		mt19937 gen(rd());
		normal_distribution<float> dist(0.0f, 0.1f);

		for (int i = 0; i < hidden_size; i++) {
			for (int j = 0; j < input_size; j++) {
				W1[i][j] = dist(gen);
			}
			b1[i] = 0.0f; 
		}

		for (int i = 0; i < output_size; i++) {
			for (int j = 0; j < hidden_size; j++) {
				W2[i][j] = dist(gen);
			}
			b2[i] = 0.0f;
		}
	}

	float sigmoid(float x) {
		return 1.0f / (1.0f + exp(-x));
	}

	float sigmoid_derivative(float x) {
		float s = sigmoid(x);
		return s * (1 - s);
	}

	float learning_rate = 0.1f;

	float mseLoss(const vector<float>& predicted,
		const vector<float>& target) {
		float sum = 0.0f;
		for (size_t i = 0; i < predicted.size(); i++) {
			float diff = predicted[i] - target[i];
			sum += diff * diff;
		}
		return sum / predicted.size();
	}
};

int main() {
	try {
		// Module creating: 2 inputs -> 4 hiddens -> 1 output (binary classification)
		LearningModule module(2, 4, 1);
		module.set_learning_rate(0.1f);

		// Adding data for XOR 
		// XOR: (0,0) -> 0, (0,1) -> 1, (1,0) -> 1, (1,1) -> 0

		// Example 1: (0,0) -> 0
		auto sample1 = make_unique<NumericDataSample>(
			vector<float>{0.0f, 0.0f},
			vector<float>{0.0f}
		);
		module.add_data_sample(move(sample1));

		// Example 2: (0,1) -> 1
		auto sample2 = std::make_unique<NumericDataSample>(
			vector<float>{0.0f, 1.0f},
			vector<float>{1.0f}
		);
		module.add_data_sample(move(sample2));

		// Example 3: (1,0) -> 1
		auto sample3 = make_unique<NumericDataSample>(
			vector<float>{1.0f, 0.0f},
			vector<float>{1.0f}
		);
		module.add_data_sample(move(sample3));

		// Example 4: (1,1) -> 0
		auto sample4 = make_unique<NumericDataSample>(
			vector<float>{1.0f, 1.0f},
		    vector<float>{0.0f}
		);
		module.add_data_sample(move(sample4));

		cout << "Starting XOR fucntion learning...\n";

		// Teaching 1000 epoches
		module.train(1000, true);

		// Checking the results out
		std::cout << "\nResults after learning:\n";

		std::vector<std::vector<float>> test_inputs = {
			{0.0f, 0.0f},
			{0.0f, 1.0f},
			{1.0f, 0.0f},
			{1.0f, 1.0f}
		};

		for (const auto& input : test_inputs) {
			auto hidden = vector<float>(4);
			auto output = vector<float>(1);

			module.forward(input, hidden, output);
			cout << input[0] << " XOR " << input[1]
				<< " = " << output[0]
				<< " (rounded: " << (output[0] > 0.5f ? 1 : 0) << ")\n";
		}

	}
	catch (const std::exception& e) {
		std::cout << "Error: " << e.what() << "\n";
	}

	return 0;
}