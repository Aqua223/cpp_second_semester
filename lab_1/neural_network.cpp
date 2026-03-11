#include "learning_module.h"

// Перемешивание датасета
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

// загрузка файла (здесь конкретная реализация под датасет)
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

        // Пропуск пустых строк
        if (line.empty()) continue;

        stringstream ss(line);
        string cell;

        vector<float> features_row;
        for (int i = 0; i < 4; i++) {
            if (!getline(ss, cell, ',')) {
                throw runtime_error("Error reading feature at line " + to_string(line_num));
            }

            // Удаление пробелов
            cell.erase(0, cell.find_first_not_of(" \t"));
            cell.erase(cell.find_last_not_of(" \t") + 1);

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

        // Чистка от кавычек, пробелов и спецсимволов
        cell.erase(remove(cell.begin(), cell.end(), '\"'), cell.end());
        cell.erase(remove(cell.begin(), cell.end(), '\''), cell.end());
        cell.erase(remove(cell.begin(), cell.end(), ' '), cell.end());
        cell.erase(remove(cell.begin(), cell.end(), '\r'), cell.end());
        cell.erase(remove(cell.begin(), cell.end(), '\n'), cell.end());

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

class NeuralNetwork {
private:
	LearningModule module;
	vector<unique_ptr<IDataSample>> training_data;
	vector<unique_ptr<IDataSample>> validation_data;

public:
	NeuralNetwork(int input_size, int hidden_size, int output_size)
		: module(input_size, hidden_size, output_size) {

		cout << "Network created!\n";
		cout << "Architecture: " << input_size << " " << hidden_size << " " << output_size << "\n";
	}

    void load_data(const string& filename, float train_ratio = 0.8f) {
        auto [features, targets] = read_file(filename);

        shuffle_dataset(features, targets);

        // Разделяем на обучающую и валидационную выборки
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

    // Обучение с валидацией
    void train_with_validation(int epochs, float learning_rate = 0.1f, int patience = 10) {
        module.set_learning_rate(learning_rate);

        float best_loss = 1e9;
        int epochs_without_improvement = 0;

        cout << "\n-START TRAINING-\n";

        for (int epoch = 0; epoch < epochs; epoch++) {
            // Обучение на одной эпохе
            float train_loss = module.train_epoch();

            // Ошибка на валидации
            float val_loss = 0;
            for (const auto& sample : validation_data) {
                auto features = sample->getFeatures();
                auto target = sample->getTarget();
                auto output = module.predict(features);
                val_loss += mse_loss(output, target);
            }
            val_loss /= validation_data.size();

            // Прогресс
            cout << "Epoch " << setw(3) << epoch + 1 << "/" << epochs
                << " | train loss: " << fixed << setprecision(4) << train_loss
                << " | val loss: " << val_loss;

            // Проверка улучшения
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

            // Ранняя остановка
            if (epochs_without_improvement >= patience) {
                cout << "Stopped. No improvements in " << patience << " epoches\n";
                break;
            }
        }

        cout << "-TRAINING COMPLETE-\n";
    }

    // Предсказание для новых данных
    vector<float> predict(const vector<float>& input) {
        auto output = module.predict(input);
        return output; // для одного выхода
    }

    // Пакетное предсказание
    vector<vector<float>> predict_batch(const vector<vector<float>>& inputs) {
        vector<vector<float>> predictions;
        for (const auto& input : inputs) {
            predictions.push_back(predict(input));
        }
        return predictions;
    }

    // Оценка модели
    void evaluate() {
        cout << "\n-ESTIMATING-\n";

        float total_loss = 0;
        int correct = 0;
        int step = 1;

        for (const auto& sample : validation_data) {
            auto features = sample->getFeatures();
            auto target = sample->getTarget();  // это вектор [1, 0, 0] или [0, 1, 0] или [0, 0, 1]
            auto output = module.predict(features);

            // Предсказанный класс
            int predicted_class = 0;
            float max_val = output[0];
            for (int i = 1; i < output.size(); i++) {
                if (output[i] > max_val) {
                    max_val = output[i];
                    predicted_class = i + 1;
                }
            }

            // Фактический класс
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

private:
    // Среднеквадратичная ошибка
    float mse_loss(const vector<float>& predicted, const vector<float>& target) {
        float sum = 0;
        for (size_t i = 0; i < predicted.size(); i++) {
            float diff = predicted[i] - target[i];
            sum += diff * diff;
        }
        return sum / predicted.size();
    }
};

// Пример использования
int main() {
    try {
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

        // Создание и обучение сети
        cout << "\nCreating neural network...\n";
        NeuralNetwork network(4, 5, 3);

        network.load_data(filename, 0.8f);
        network.train_with_validation(100, 0.1f, 10);
        network.evaluate();

        auto predictions = network.predict_batch(features);
        int correct = 0;
        int step = 1; 

        cout << "\n-ANOTHER TESTING-\n";

        for (int i = 0; i < predictions.size(); i++) {
            cout << step++ << ". ";
            for (float p : predictions[i]) {
                cout << p << ' ';
            }
            int predicted_class = 1;
            float max_val = predictions[i][0];
            for (int j = 1; j < predictions[i].size(); j++) {
                if (predictions[i][j] > max_val) {
                    max_val = predictions[i][j];
                    predicted_class = j + 1;
                }
            }

            int actual_class = 1;
            for (int j = 0; j < targets[i].size(); j++) {
                if (targets[i][j]) {
                    actual_class = j + 1;
                    break;
                }
            }

            cout << "| Predicted: " << predicted_class << ". Actual: " << actual_class << ' ';
            if (predicted_class == actual_class) {
                cout << "| Correct!";
                correct += 1;
            }

            cout << '\n';
        }
        cout << "\nAccuracy: " << setprecision(2) << 1.0f * correct / predictions.size() * 100 << "%\n";
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
    }
}