#include "learning_module.h"

using namespace std;

// Класс NeuralNetwork для бинарной классификации
class NeuralNetwork {
private:
    LearningModule module;
    vector<unique_ptr<IDataSample>> validation_data;

    float binary_cross_entropy(const vector<float>& predicted, const vector<float>& target);
    float accuracy_score(const vector<float>& predicted, const vector<float>& target);

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);

    void generate_circle_data(int num_samples, float train_ratio = 0.8f);
    void generate_linear_data(int num_samples, float train_ratio = 0.8f);
    void generate_xor_data(int num_samples, float train_ratio = 0.8f);

    void train_with_validation(int epochs, float learning_rate = 0.1f, int patience = 10);

    vector<float> predict(const vector<float>& input);
    int predict_class(const vector<float>& input);
    vector<vector<float>> predict_batch(const vector<vector<float>>& inputs);

    void evaluate();
    void visualize_decision_boundary();
};

// Реализация NumericDataSample

NumericDataSample::NumericDataSample(const vector<float>& f, const vector<float>& t)
    : features(f), target(t) {
}

vector<float> NumericDataSample::getFeatures() const {
    return features;
}

vector<float> NumericDataSample::getTarget() const {
    return target;
}

size_t NumericDataSample::getFeatureSize() const {
    return features.size();
}

size_t NumericDataSample::getTargetSize() const {
    return target.size();
}

// Реализация LearningModule

LearningModule::LearningModule(int input_size, int hidden_size, int output_size)
    : input_size(input_size),
    hidden_size(hidden_size),
    output_size(output_size) {

    W1.resize(hidden_size, vector<float>(input_size));
    W2.resize(output_size, vector<float>(hidden_size));

    b1.resize(hidden_size);
    b2.resize(output_size);

    initialize_weights();

    cout << "Module created!\n";
    cout << "Architecture: " << input_size << " - " << hidden_size << " - " << output_size << '\n';
}

void LearningModule::initialize_weights() {
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

float LearningModule::sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float LearningModule::sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1 - s);
}

float LearningModule::bce_loss(const vector<float>& predicted, const vector<float>& target) {
    float eps = 1e-7f;
    float loss = 0.0f;
    for (size_t i = 0; i < predicted.size(); i++) {
        float p = max(min(predicted[i], 1.0f - eps), eps);
        float t = target[i];
        loss += -(t * log(p) + (1 - t) * log(1 - p));
    }
    return loss / predicted.size();
}

void LearningModule::add_data_sample(unique_ptr<IDataSample> sample) {
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

void LearningModule::forward(const vector<float>& input,
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

void LearningModule::backward(const vector<float>& input,
    const vector<float>& target,
    const vector<float>& hidden,
    const vector<float>& output) {

    vector<float> grad_output(output_size);
    for (int i = 0; i < output_size; i++) {
        grad_output[i] = output[i] - target[i];
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

void LearningModule::set_learning_rate(float lr) {
    learning_rate = lr;
}

float LearningModule::train_step(const vector<float>& input,
    const vector<float>& target) {

    auto hidden = vector<float>(hidden_size);
    auto output = vector<float>(output_size);

    forward(input, hidden, output);

    float loss = bce_loss(output, target);

    backward(input, target, hidden, output);

    return loss;
}

float LearningModule::train_epoch() {
    float totalLoss = 0.0f;

    for (const auto& sample : dataset) {
        auto features = sample->getFeatures();
        auto target = sample->getTarget();

        float loss = train_step(features, target);
        totalLoss += loss;
    }

    return totalLoss / dataset.size();
}

void LearningModule::train(int epochs, bool verbose) {
    if (dataset.empty()) {
        cout << "No data for training!\n";
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        float avgLoss = train_epoch();

        if (verbose && (epoch + 1) % 10 == 0) {
            cout << "Epoch " << epoch + 1 << '/' << epochs
                << ", mean loss: " << avgLoss << '\n';
        }
    }
}

vector<float> LearningModule::predict(const vector<float>& input) {
    vector<float> hidden(hidden_size);
    vector<float> output(output_size);
    forward(input, hidden, output);
    return output;
}

float LearningModule::evaluate(const vector<unique_ptr<IDataSample>>& test_data) {
    float total_loss = 0;
    for (const auto& sample : test_data) {
        auto features = sample->getFeatures();
        auto target = sample->getTarget();
        auto output = predict(features);
        total_loss += bce_loss(output, target); 
    }
    return total_loss / test_data.size();
}

// Реализация NeuralNetwork

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : module(input_size, hidden_size, output_size) {

    cout << "Binary Classification Network created!\n";
    cout << "Architecture: " << input_size << " - " << hidden_size << " - " << output_size << "\n";
}

float NeuralNetwork::binary_cross_entropy(const vector<float>& predicted, const vector<float>& target) {
    float eps = 1e-7f;
    float p = max(min(predicted[0], 1.0f - eps), eps);
    float t = target[0];
    return -(t * log(p) + (1 - t) * log(1 - p));
}

float NeuralNetwork::accuracy_score(const vector<float>& predicted, const vector<float>& target) {
    int pred_class = (predicted[0] >= 0.5f) ? 1 : 0;
    int true_class = (target[0] >= 0.5f) ? 1 : 0;
    return (pred_class == true_class) ? 1.0f : 0.0f;
}

void NeuralNetwork::generate_circle_data(int num_samples, float train_ratio) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-2.0f, 2.0f);

    vector<vector<float>> features;
    vector<vector<float>> targets;

    for (int i = 0; i < num_samples; i++) {
        float x = dist(gen);
        float y = dist(gen);
        float label = (sqrt(x * x + y * y) <= 1.0f) ? 1.0f : 0.0f;

        features.push_back({ x, y });
        targets.push_back({ label });
    }

    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++) indices[i] = i;

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    size_t train_size = static_cast<size_t>(num_samples * train_ratio);

    for (int i = 0; i < num_samples; i++) {
        int idx = indices[i];
        auto sample = make_unique<NumericDataSample>(features[idx], targets[idx]);

        if (i < train_size) {
            module.add_data_sample(move(sample));
        }
        else {
            validation_data.push_back(move(sample));
        }
    }

    cout << "Generated circle data: overall=" << num_samples
        << ", training=" << train_size
        << ", validation=" << num_samples - train_size << "\n";
    cout << "Task: Classify points inside (class 1) or outside (class 0) circle radius 1\n";
}

void NeuralNetwork::generate_linear_data(int num_samples, float train_ratio) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-2.0f, 2.0f);

    vector<vector<float>> features;
    vector<vector<float>> targets;

    for (int i = 0; i < num_samples; i++) {
        float x = dist(gen);
        float y = dist(gen);
        float label = (x + y > 0) ? 1.0f : 0.0f;

        features.push_back({ x, y });
        targets.push_back({ label });
    }

    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++) indices[i] = i;

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    size_t train_size = static_cast<size_t>(num_samples * train_ratio);

    for (int i = 0; i < num_samples; i++) {
        int idx = indices[i];
        auto sample = make_unique<NumericDataSample>(features[idx], targets[idx]);

        if (i < train_size) {
            module.add_data_sample(move(sample));
        }
        else {
            validation_data.push_back(move(sample));
        }
    }

    cout << "Generated linear data: overall=" << num_samples
        << ", training=" << train_size
        << ", validation=" << num_samples - train_size << "\n";
    cout << "Task: Classify points where x + y > 0 (class 1) or x + y <= 0 (class 0)\n";
}

void NeuralNetwork::generate_xor_data(int num_samples, float train_ratio) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<float> dist(-2.0f, 2.0f);

    vector<vector<float>> features;
    vector<vector<float>> targets;

    for (int i = 0; i < num_samples; i++) {
        float x = dist(gen);
        float y = dist(gen);
        float label = ((x > 0) != (y > 0)) ? 1.0f : 0.0f;

        features.push_back({ x, y });
        targets.push_back({ label });
    }

    vector<int> indices(num_samples);
    for (int i = 0; i < num_samples; i++) indices[i] = i;

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    size_t train_size = static_cast<size_t>(num_samples * train_ratio);

    for (int i = 0; i < num_samples; i++) {
        int idx = indices[i];
        auto sample = make_unique<NumericDataSample>(features[idx], targets[idx]);

        if (i < train_size) {
            module.add_data_sample(move(sample));
        }
        else {
            validation_data.push_back(move(sample));
        }
    }

    cout << "Generated XOR data: overall=" << num_samples
        << ", training=" << train_size
        << ", validation=" << num_samples - train_size << "\n";
    cout << "Task: XOR classification (class 1 if x and y have different signs)\n";
}

void NeuralNetwork::train_with_validation(int epochs, float learning_rate, int patience) {
    module.set_learning_rate(learning_rate);

    float best_loss = 1e9;
    float best_accuracy = 0.0f;
    int epochs_without_improvement = 0;

    cout << "\n-START TRAINING-\n";
    cout << "Learning rate: " << learning_rate << "\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        float train_loss = module.train_epoch();

        float val_loss = 0;
        float val_accuracy = 0;

        for (const auto& sample : validation_data) {
            auto features = sample->getFeatures();
            auto target = sample->getTarget();
            auto output = module.predict(features);
            val_loss += binary_cross_entropy(output, target);
            val_accuracy += accuracy_score(output, target);
        }
        val_loss /= validation_data.size();
        val_accuracy /= validation_data.size();

        cout << "Epoch " << setw(3) << epoch + 1 << "/" << epochs
            << " | train loss: " << fixed << setprecision(4) << train_loss
            << " | val loss: " << val_loss
            << " | val acc: " << setprecision(2) << val_accuracy * 100 << "%";

        if (val_loss < best_loss) {
            best_loss = val_loss;
            best_accuracy = val_accuracy;
            epochs_without_improvement = 0;
            cout << " (best result)";
        }
        else {
            epochs_without_improvement++;
            cout << " (" << epochs_without_improvement << "/" << patience << ")";
        }

        cout << "\n";

        if (epochs_without_improvement >= patience) {
            cout << "Early stopping! No improvement in " << patience << " epochs\n";
            break;
        }
    }

    cout << "-TRAINING COMPLETE-\n";
    cout << "Best validation loss: " << fixed << setprecision(4) << best_loss << "\n";
    cout << "Best validation accuracy: " << best_accuracy * 100 << "%\n";
}

vector<float> NeuralNetwork::predict(const vector<float>& input) {
    return module.predict(input);
}

int NeuralNetwork::predict_class(const vector<float>& input) {
    auto output = module.predict(input);
    return (output[0] >= 0.5f) ? 1 : 0;
}

vector<vector<float>> NeuralNetwork::predict_batch(const vector<vector<float>>& inputs) {
    vector<vector<float>> predictions;
    for (const auto& input : inputs) {
        predictions.push_back(predict(input));
    }
    return predictions;
}

void NeuralNetwork::evaluate() {
    cout << "\n-EVALUATION ON VALIDATION SET-\n";

    float total_loss = 0;
    int correct = 0;
    int true_positives = 0, false_positives = 0;
    int true_negatives = 0, false_negatives = 0;
    int step = 1;

    cout << "Detailed predictions (first 20 samples):\n";
    cout << "----------------------------------------\n";

    for (const auto& sample : validation_data) {
        auto features = sample->getFeatures();
        auto target = sample->getTarget();
        auto output = module.predict(features);

        int predicted_class = (output[0] >= 0.5f) ? 1 : 0;
        int actual_class = (target[0] >= 0.5f) ? 1 : 0;

        if (predicted_class == actual_class) {
            correct++;
        }

        if (predicted_class == 1 && actual_class == 1) true_positives++;
        else if (predicted_class == 1 && actual_class == 0) false_positives++;
        else if (predicted_class == 0 && actual_class == 1) false_negatives++;
        else if (predicted_class == 0 && actual_class == 0) true_negatives++;

        total_loss += binary_cross_entropy(output, target);

        if (step <= 20) {
            cout << step << ". Point (" << fixed << setprecision(2)
                << features[0] << ", " << features[1] << ")"
                << " | Output: " << setprecision(4) << output[0]
                << " | Predicted: " << predicted_class
                << " | Actual: " << actual_class;

            if (predicted_class == actual_class) {
                cout << " +";
            }
            cout << "\n";
        }
        step++;
    }

    float accuracy = 100.0f * correct / validation_data.size();
    float avg_loss = total_loss / validation_data.size();

    float precision = (true_positives + false_positives > 0) ?
        100.0f * true_positives / (true_positives + false_positives) : 0;
    float recall = (true_positives + false_negatives > 0) ?
        100.0f * true_positives / (true_positives + false_negatives) : 0;
    float f1_score = (precision + recall > 0) ?
        2 * precision * recall / (precision + recall) : 0;

    cout << "\n----------------------------------------\n";
    cout << "Confusion Matrix:\n";
    cout << "               Predicted\n";
    cout << "              0       1\n";
    cout << "Actual   0   " << setw(4) << true_negatives << "    " << setw(4) << false_positives << "\n";
    cout << "         1   " << setw(4) << false_negatives << "    " << setw(4) << true_positives << "\n\n";

    cout << "Accuracy:  " << fixed << setprecision(2) << accuracy << "% (" << correct << "/" << validation_data.size() << ")\n";
    cout << "Precision: " << precision << "%\n";
    cout << "Recall:    " << recall << "%\n";
    cout << "F1-Score:  " << f1_score << "%\n";
    cout << "BCE Loss:  " << avg_loss << "\n";
}

void NeuralNetwork::visualize_decision_boundary() {
    cout << "\n-DECISION BOUNDARY VISUALIZATION-\n";
    cout << "Grid of predictions ('.' = class 0, '#' = class 1):\n\n";

    for (float y = 2.0f; y >= -2.0f; y -= 0.15f) {
        for (float x = -2.0f; x <= 2.0f; x += 0.1f) {
            int pred = predict_class({ x, y });
            cout << (pred == 1 ? '#' : '.');
        }
        cout << "\n";
    }
    cout << "\nLegend: . = Class 0, # = Class 1\n";
}

// Тестирование модели для бинарной классификации

int main() {
    try {
        cout << "========================================\n";
        cout << "BINARY CLASSIFICATION ON 2D PLANE\n";
        cout << "========================================\n";

        cout << "\nChoose classification type:\n";
        cout << "1. Circle (inside/outside radius 1)\n";
        cout << "2. Linear (x + y > 0)\n";
        cout << "3. XOR (different signs)\n";
        cout << "Enter choice (1-3): ";

        int choice;
        cin >> choice;

        cout << "\nCreating neural network...\n";
        NeuralNetwork network(2, 8, 1);

        switch (choice) {
        case 1:
            cout << "\nGenerating circle classification data...\n";
            network.generate_circle_data(500, 0.8f);
            break;
        case 2:
            cout << "\nGenerating linear classification data...\n";
            network.generate_linear_data(500, 0.8f);
            break;
        case 3:
            cout << "\nGenerating XOR classification data...\n";
            network.generate_xor_data(500, 0.8f);
            break;
        default:
            cout << "Invalid choice, using circle classification\n";
            network.generate_circle_data(500, 0.8f);
        }

        network.train_with_validation(100, 0.2f, 15);
        network.evaluate();
        network.visualize_decision_boundary();

        cout << "\n-TESTING ON SPECIFIC POINTS-\n";

        if (choice == 1) {
            cout << "Point (0.0, 0.0) -> Class " << network.predict_class({ 0.0f, 0.0f }) << " (expected: 1 - inside circle)\n";
            cout << "Point (1.5, 0.0) -> Class " << network.predict_class({ 1.5f, 0.0f }) << " (expected: 0 - outside circle)\n";
            cout << "Point (0.5, 0.5) -> Class " << network.predict_class({ 0.5f, 0.5f }) << " (expected: 1 - inside circle)\n";
            cout << "Point (1.2, 1.2) -> Class " << network.predict_class({ 1.2f, 1.2f }) << " (expected: 0 - outside circle)\n";
            cout << "Point (-0.8, -0.6) -> Class " << network.predict_class({ -0.8f, -0.6f }) << " (expected: 1 - inside circle)\n";
        }
        else if (choice == 2) {
            cout << "Point (1.0, 1.0) -> Class " << network.predict_class({ 1.0f, 1.0f }) << " (expected: 1 - x+y>0)\n";
            cout << "Point (-1.0, -1.0) -> Class " << network.predict_class({ -1.0f, -1.0f }) << " (expected: 0 - x+y<0)\n";
            cout << "Point (1.0, -0.5) -> Class " << network.predict_class({ 1.0f, -0.5f }) << " (expected: 1 - x+y>0)\n";
            cout << "Point (-1.0, 0.5) -> Class " << network.predict_class({ -1.0f, 0.5f }) << " (expected: 0 - x+y<0)\n";
        }
        else {
            cout << "Point (1.0, 1.0) -> Class " << network.predict_class({ 1.0f, 1.0f }) << " (expected: 0 - same signs)\n";
            cout << "Point (-1.0, -1.0) -> Class " << network.predict_class({ -1.0f, -1.0f }) << " (expected: 0 - same signs)\n";
            cout << "Point (1.0, -1.0) -> Class " << network.predict_class({ 1.0f, -1.0f }) << " (expected: 1 - different signs)\n";
            cout << "Point (-1.0, 1.0) -> Class " << network.predict_class({ -1.0f, 1.0f }) << " (expected: 1 - different signs)\n";
        }

    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}