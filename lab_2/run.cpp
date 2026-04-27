#include "neural_network.h"

int main() {
    try {
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