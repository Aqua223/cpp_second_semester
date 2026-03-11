#pragma once

#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>
#include <random>
#include <chrono>
#include <algorithm>

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
	int get_input_size() const { return input_size; }
	int get_hidden_size() const { return hidden_size; }
	int get_output_size() const { return output_size; }

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

	void set_learning_rate(float lr) {
		learning_rate = lr;
	}

	float train_step(const vector<float>& input,
		const vector<float>& target) {

		auto hidden = vector<float>(hidden_size);
		auto output = vector<float>(output_size);

		forward(input, hidden, output);

		float loss = mse_loss(output, target);

		backward(input, target, hidden, output);

		return loss;
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
			cout << "No data for training!\n";
			return;
		}

		for (int epoch = 0; epoch < epochs; epoch++) {
			float avgLoss = train_epoch();

			if (verbose) {
				cout << "Epoch " << epoch + 1 << '/' << epochs
					<< ", mean loss: " << avgLoss << '\n';
			}
		}
	}

	vector<float> predict(const vector<float>& input) {
		vector<float> hidden(hidden_size);
		vector<float> output(output_size);
		forward(input, hidden, output);
		return output;
	}

	float evaluate(const vector<unique_ptr<IDataSample>>& test_data) {
		float total_loss = 0;
		for (const auto& sample : test_data) {
			auto features = sample->getFeatures();
			auto target = sample->getTarget();
			auto output = predict(features);
			total_loss += mse_loss(output, target);
		}
		return total_loss / test_data.size();
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

	float mse_loss(const vector<float>& predicted,
		const vector<float>& target) {
		float sum = 0.0f;
		for (size_t i = 0; i < predicted.size(); i++) {
			float diff = predicted[i] - target[i];
			sum += diff * diff;
		}
		return sum / predicted.size();
	}
};