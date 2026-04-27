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
    NumericDataSample(const vector<float>& f, const vector<float>& t);

    vector<float> getFeatures() const override;
    vector<float> getTarget() const override;
    size_t getFeatureSize() const override;
    size_t getTargetSize() const override;
};