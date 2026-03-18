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
    NumericDataSample(const vector<float>& f, const vector<float>& t);

    vector<float> getFeatures() const override;
    vector<float> getTarget() const override;
    size_t getFeatureSize() const override;
    size_t getTargetSize() const override;
};

class LearningModule {
private:
    int input_size;
    int hidden_size;
    int output_size;
    float learning_rate = 0.1f;

    vector<vector<float>> W1, W2;
    vector<float> b1, b2;

    vector<unique_ptr<IDataSample>> dataset;

    void initialize_weights();

    float sigmoid(float x);

    float sigmoid_derivative(float x);

    float mse_loss(const vector<float>& predicted, const vector<float>& target);

public:
    LearningModule(int input_size, int hidden_size, int output_size);

    int get_input_size() const { return input_size; }
    int get_hidden_size() const { return hidden_size; }
    int get_output_size() const { return output_size; }

    void add_data_sample(unique_ptr<IDataSample> sample);

    void forward(const vector<float>& input, vector<float>& hidden, vector<float>& output);

    void backward(const vector<float>& input, const vector<float>& target,
        const vector<float>& hidden, const vector<float>& output);

    void set_learning_rate(float lr);

    float train_step(const vector<float>& input, const vector<float>& target);

    float train_epoch();

    void train(int epochs, bool verbose = true);

    vector<float> predict(const vector<float>& input);

    float evaluate(const vector<unique_ptr<IDataSample>>& test_data);
};