#include "learning_module.h"

using namespace std;

// Инициализация NeuralNetwork

class NeuralNetwork {
private:
    LearningModule module;
    vector<unique_ptr<IDataSample>> training_data;
    vector<unique_ptr<IDataSample>> validation_data;

    float mse_loss(const vector<float>& predicted, const vector<float>& target);

public:
    NeuralNetwork(int input_size, int hidden_size, int output_size);

    void load_data(const string& filename, float train_ratio = 0.8f);
    void train_with_validation(int epochs, float learning_rate = 0.1f, int patience = 10);
    vector<float> predict(const vector<float>& input);
    vector<vector<float>> predict_batch(const vector<vector<float>>& inputs);
    void evaluate();
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
    cout << "Architecture: " << input_size << ' ' << hidden_size << ' ' << output_size << '\n';
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

float LearningModule::mse_loss(const vector<float>& predicted, const vector<float>& target) {
    float sum = 0.0f;
    for (size_t i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
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

void LearningModule::set_learning_rate(float lr) {
    learning_rate = lr;
}

float LearningModule::train_step(const vector<float>& input,
    const vector<float>& target) {

    auto hidden = vector<float>(hidden_size);
    auto output = vector<float>(output_size);

    forward(input, hidden, output);

    float loss = mse_loss(output, target);

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

        if (verbose) {
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
        total_loss += mse_loss(output, target);
    }
    return total_loss / test_data.size();
}

// Вспомогательные функции

string clean_string(string str) {
    str.erase(remove(str.begin(), str.end(), '\"'), str.end());
    str.erase(remove(str.begin(), str.end(), '\''), str.end());
    str.erase(remove(str.begin(), str.end(), ' '), str.end());
    str.erase(remove(str.begin(), str.end(), '\r'), str.end());
    str.erase(remove(str.begin(), str.end(), '\n'), str.end());
    return str;
}

string clean_digit(string str) {
    str.erase(0, str.find_first_not_of(" \t"));
    str.erase(str.find_last_not_of(" \t") + 1);

    return str;
}

void shuffle_dataset(vector<vector<float>>& features, vector<vector<float>>& targets) {
    vector<int> indices(features.size());
    for (int i = 0; i < features.size(); i++) {
        indices[i] = i;
    }

    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    shuffle(indices.begin(), indices.end(), default_random_engine(seed));

    vector<vector<float>> shuffled_features;
    vector<vector<float>> shuffled_targets;

    for (int idx : indices) {
        shuffled_features.push_back(features[idx]);
        shuffled_targets.push_back(targets[idx]);
    }

    features = move(shuffled_features);
    targets = move(shuffled_targets);
}

pair<vector<vector<float>>, vector<vector<float>>> read_file(const string& filename) {
    vector<vector<float>> features;
    vector<vector<float>> targets;

    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + filename);
    }

    string line;
    int line_num = 0;

    while (getline(file, line)) {
        line_num++;

        if (line.empty()) continue;

        stringstream ss(line);
        string cell;

        vector<float> features_row;
        for (int i = 0; i < 4; i++) {
            if (!getline(ss, cell, ',')) {
                throw runtime_error("Error reading feature at line " + to_string(line_num));
            }

            // для очистки признаков от лишних пробелов и табуляции по краям
            cell = clean_digit(cell);

            try {
                features_row.push_back(stof(cell));
            }
            catch (...) {
                throw runtime_error("Invalid number at line " + to_string(line_num) + ": " + cell);
            }
        }
        features.push_back(features_row);

        if (!getline(ss, cell, ',')) {
            throw runtime_error("Missing class at line " + to_string(line_num));
        }

        // Обработка названия класса цветка (лишние кавычки, пробелы и т.д.)
        cell = clean_string(cell);

        vector<float> target_row(3, 0.0f);

        if (cell == "Iris-setosa" || cell == "setosa") {
            target_row[0] = 1.0f;
        }
        else if (cell == "Iris-versicolor" || cell == "versicolor") {
            target_row[1] = 1.0f;
        }
        else if (cell == "Iris-virginica" || cell == "virginica") {
            target_row[2] = 1.0f;
        }
        else {
            cout << "Warning: Unknown class '" << cell << "' at line " << line_num << endl;
            features.pop_back();
            continue;
        }

        targets.push_back(target_row);
    }

    file.close();

    cout << "Successfully loaded " << features.size() << " samples from " << filename << endl;
    if (!features.empty()) {
        cout << "Features dimension: " << features[0].size() << endl;
        cout << "Output dimension: " << targets[0].size() << " (one-hot encoding)" << endl;
    }

    return { features, targets };
}

// Реализация NeuralNetwork

NeuralNetwork::NeuralNetwork(int input_size, int hidden_size, int output_size)
    : module(input_size, hidden_size, output_size) {

    cout << "Network created!\n";
    cout << "Architecture: " << input_size << " " << hidden_size << " " << output_size << "\n";
}

void NeuralNetwork::load_data(const string& filename, float train_ratio) {
    auto [features, targets] = read_file(filename);

    shuffle_dataset(features, targets);

    size_t train_size = static_cast<size_t>(features.size() * train_ratio);

    for (size_t i = 0; i < features.size(); i++) {
        auto sample = make_unique<NumericDataSample>(features[i], targets[i]);

        if (i < train_size) {
            module.add_data_sample(move(sample));
        }
        else {
            validation_data.push_back(move(sample));
        }
    }

    cout << "Data loaded: overall=" << features.size()
        << ", training=" << train_size
        << ", validated=" << features.size() - train_size << "\n";
}

void NeuralNetwork::train_with_validation(int epochs, float learning_rate, int patience) {
    module.set_learning_rate(learning_rate);

    float best_loss = 1e9;
    int epochs_without_improvement = 0;

    cout << "\n-START TRAINING-\n";

    for (int epoch = 0; epoch < epochs; epoch++) {
        float train_loss = module.train_epoch();

        float val_loss = 0;
        for (const auto& sample : validation_data) {
            auto features = sample->getFeatures();
            auto target = sample->getTarget();
            auto output = module.predict(features);
            val_loss += mse_loss(output, target);
        }
        val_loss /= validation_data.size();

        cout << "Epoch " << setw(3) << epoch + 1 << "/" << epochs
            << " | train loss: " << fixed << setprecision(4) << train_loss
            << " | val loss: " << val_loss;

        if (val_loss < best_loss) {
            best_loss = val_loss;
            epochs_without_improvement = 0;
            cout << " (best result)";
        }
        else {
            epochs_without_improvement++;
            cout << " (" << epochs_without_improvement << "/" << patience << ")";
        }

        cout << "\n";

        if (epochs_without_improvement >= patience) {
            cout << "Stopped. No improvements in " << patience << " epoches\n";
            break;
        }
    }

    cout << "-TRAINING COMPLETE-\n";
}

vector<float> NeuralNetwork::predict(const vector<float>& input) {
    return module.predict(input);
}

vector<vector<float>> NeuralNetwork::predict_batch(const vector<vector<float>>& inputs) {
    vector<vector<float>> predictions;
    for (const auto& input : inputs) {
        predictions.push_back(predict(input));
    }
    return predictions;
}

float NeuralNetwork::mse_loss(const vector<float>& predicted, const vector<float>& target) {
    float sum = 0;
    for (size_t i = 0; i < predicted.size(); i++) {
        float diff = predicted[i] - target[i];
        sum += diff * diff;
    }
    return sum / predicted.size();
}

void NeuralNetwork::evaluate() {
    cout << "\n-ESTIMATING-\n";

    float total_loss = 0;
    int correct = 0;
    int step = 1;

    for (const auto& sample : validation_data) {
        auto features = sample->getFeatures();
        auto target = sample->getTarget();
        auto output = module.predict(features);

        int predicted_class = 0;
        float max_val = output[0];
        for (int i = 1; i < output.size(); i++) {
            if (output[i] > max_val) {
                max_val = output[i];
                predicted_class = i + 1;
            }
        }

        int actual_class = 0;
        for (int i = 0; i < target.size(); i++) {
            if (target[i]) {
                actual_class = i + 1;
                break;
            }
        }

        if (predicted_class == actual_class) {
            correct++;
        }

        total_loss += mse_loss(output, target);

        cout << step++ << ". Predicted: " << predicted_class << "; expected: " << actual_class << '\n';
    }

    float accuracy = 100.0f * correct / validation_data.size();
    float avg_loss = total_loss / validation_data.size();

    cout << "Accuracy: " << fixed << setprecision(2) << accuracy
        << "% (" << correct << "/" << validation_data.size() << ")\n";
    cout << "MSE: " << avg_loss << "\n";
}

// Тестирование модели

int main() {
    try {
        // Очень простое пользование моделью на примере датасета IRIS
        cout << "Current directory contents:\n";

        string filename = "iris.txt";

        cout << "\nTrying to read " << filename << "...\n\n";

        auto [features, targets] = read_file(filename);

        cout << "\nFirst 5 samples:\n";
        for (int i = 0; i < min(5, (int)features.size()); i++) {
            cout << "Features: ";
            for (float f : features[i]) {
                cout << f << " ";
            }
            cout << "-> Target: ";
            for (float t : targets[i]) {
                cout << t << " ";
            }
            cout << '\n';
        }

        cout << "\nCreating neural network...\n";
        NeuralNetwork network(4, 5, 3);

        network.load_data(filename, 0.8f);
        network.train_with_validation(100, 0.1f, 10);
        network.evaluate();
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}