#include <cmath>
#include <iostream>

#include <boost/algorithm/string.hpp>

#include <curl/curl.h>

#include "SymReg/SymbolicRegressor.h"

using namespace sr;

using Result = sr::SymbolicRegressor<double>::Result;

struct BenchmarkResult
{
    std::string name;
    std::string expression;
    Result result;
};

size_t writeCallback(void* contents, size_t size, size_t nmemb, void* userp)
{
    size_t totalSize = size * nmemb;
    std::string* str = static_cast<std::string*>(userp);
    str->append(static_cast<char*>(contents), totalSize);

    return totalSize;
}

std::string downloadUrlToString(std::string const& url)
{
    CURL* curl;
    CURLcode res;
    std::string result;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl)
    {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &result);

        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;

        curl_easy_cleanup(curl);
    }

    curl_global_cleanup();

    return result;
}

struct StrotagtzData
{
    Eigen::Array<double, Eigen::Dynamic, 1> label;
    Eigen::Array<double, Eigen::Dynamic, 1> x;
    Eigen::Array<double, Eigen::Dynamic, 1> y;
};

StrotagtzData downloadUrlData(std::string const& url)
{
    auto const content{downloadUrlToString(url)};

    std::vector<std::string> lines;
    boost::split(lines, content, boost::is_any_of("\n"));

    std::vector<double> label, x, y;

    for (size_t i{1}; i < lines.size(); ++i)
    {
        if (lines[i].empty())
            continue;

        std::vector<std::string> values;
        boost::split(values, lines[i], boost::is_any_of(","));

        label.emplace_back(std::stod(values[0]));
        x.emplace_back(std::stod(values[1]));
        y.emplace_back(std::stod(values[2]));
    }

    StrotagtzData data;

    data.label.resize(label.size());
    data.x.resize(x.size());
    data.y.resize(y.size());

    for (size_t i{0}; i < label.size(); ++i)
    {
        data.label[i] = label[i];
        data.x[i] = x[i];
        data.y[i] = y[i];
    }

    return data;
}

Eigen::ArrayXd noiseData(Eigen::ArrayXd const& y, double percentage)
{
    assert(0.0 <= percentage && percentage <= 100.0);
    
    auto const ratio{percentage / 100.0};

    Eigen::ArrayXd z{y};
    z.setRandom();

    return y + ratio * z * y;
}

double bestLoss = std::numeric_limits<double>::infinity();

void callback(SymbolicRegressor<double>::Result const& result)
{
    if (result.loss < bestLoss)
    {
        bestLoss = result.loss;
        std::cout << expr(result.expression.str()) << std::endl;
        std::cout << result.expression.optStr() << std::endl;
    }
}

BenchmarkResult test_5x1Add7x2Addx3Add8(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 = noiseData(x1, noisePercentage);
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 = noiseData(x2, noisePercentage);
    Eigen::ArrayXd x3(n);
    x3.setRandom();
    x3 = noiseData(x3, noisePercentage);

    auto const y{5.2 * x1 + 7.3 * x2 + x3 + 8.6};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    auto const ls{Eigen::ArrayXd::LinSpaced(11, 0, 10)};
    std::vector<double> paramsValue;
    //for (int i{0}; i < ls.size(); ++i)
    //    paramsValue.emplace_back(ls[i]);

    std::map<std::string, size_t> operatorDepth;
    operatorDepth["+"] = 2;

    Expression<double> e{BinaryOperator<double>::plus(),
                         Variable<double>{"x1", x1},
                         Expression<double>{BinaryOperator<double>::plus(),
                                            Variable<double>{"x3", x3}, Variable<double>{"x2", x2}}};
    std::vector<double> params;
    e.params(params);
    params = std::vector<double>{0, 1, 1, 1};
    e.applyParams(params);
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(e);

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2), Var("x3", x3)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    return BenchmarkResult{"5x1Add7x2Addx3Add8",
                           "5.2*x1+7.3*x2+x3+8.6",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_LinearFit(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    Eigen::ArrayXd x(7);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    using Var = Variable<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)}};

    auto const y(2 * x + 3);

    return BenchmarkResult{"LinearFit",
                           "2*x+3",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_LogFit(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    Eigen::ArrayXd x(100);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    auto const ls{Eigen::ArrayXd::LinSpaced(5 + 1, 0, 5)};
    std::vector<double> paramsValue;
    for (int i{0}; i < ls.size(); ++i)
        paramsValue.emplace_back(ls[i]);

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::log()},
                         std::vector<BinOp>{},
                         1,
                         paramsValue};

    auto const y(2 * Eigen::log(3 * x + 4) + 5);

    return BenchmarkResult{"LogFit",
                           "2*log(3*x+4)+5",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Test1(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;
    x = noiseData(x, noisePercentage);

    auto const y{x.pow(2) + x + 1};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times()},
                         3};

    return BenchmarkResult{"Test1",
                           "x**2+x+1",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Test2(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::sin(x) * Eigen::exp(x)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    int constexpr min{-2 + 0};
    int constexpr max{2 - 1 + 1};
    auto const ls{Eigen::ArrayXd::LinSpaced(max - min + 1, min, max)};
    std::vector<double> paramsValue;
    for (int i{0}; i < ls.size(); ++i)
        paramsValue.emplace_back(ls[i]);
    //paramsValue = std::vector<double>{0, 1};

    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>(BinOp::times(),
    //                                                 Expression<double>(UnOp::sin(), Var("x", x)),
    //                                                 Expression<double>(UnOp::exp(), Var("x", x))));

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sin(), UnOp::exp()},
                         std::vector<BinOp>{BinOp::times()},
                         1,
                         paramsValue,
                         std::map<std::string, size_t>{},
                         extraExpressions};

    return BenchmarkResult{"Test2",
                           "exp(x)*sin(x)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Test3(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;
    x = noiseData(x, noisePercentage);

    auto const y{x / (1 + x * x)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>(BinOp::divide(),
    //                                                 Var("x", x),
    //                                                 Expression<double>(BinOp::times(),
    //                                                                    Var("x", x),
    //                                                                    Var("x", x))));
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["/"] = 1;
    operatorDepth["*"] = 1;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::divide()},
                         2,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    return BenchmarkResult{"Test3",
                           "x / (1 + x**2)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Test4(double noisePercentage)
{
    //x**2+y**2

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);
    Eigen::ArrayXd y(n);
    y.setRandom();
    y = noiseData(y, noisePercentage);

    auto const z{x.pow(2) + y.pow(2)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x), Var("y", y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3};

    return BenchmarkResult{"Test4",
                           "x**2+y**2",
                           sr.fit(noiseData(z, noisePercentage))};
}

BenchmarkResult test_Test5(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;
    x += 11;
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::log(x) + Eigen::sin(x)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sin(), UnOp::log()},
                         std::vector<BinOp>{BinOp::plus()},
                         2,
                         paramsValue};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"Test5",
                           "log(x)+sin(x)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Test6(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::exp(-x * x / 2) / std::sqrt(2 * M_PI)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, -0.5, 1.0 / std::sqrt(2 * M_PI)};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::exp()},
                         std::vector<BinOp>{BinOp::times()},
                         3,
                         paramsValue};
    sr.keepBestLimit = 25;

    return BenchmarkResult{"Test6",
                           "exp(-0.5*x**2)/sqrt(2*pi)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_PySR(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x0(n);
    x0.setRandom();
    x0 = noiseData(x0, noisePercentage);
    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 = noiseData(x1, noisePercentage);
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 = noiseData(x2, noisePercentage);
    Eigen::ArrayXd x3(n);
    x3.setRandom();
    x3 = noiseData(x3, noisePercentage);
    Eigen::ArrayXd x4(n);
    x4.setRandom();
    x4 = noiseData(x4, noisePercentage);

    auto const y{2.5382 * Eigen::cos(x3) + x0 * x0 - 0.5};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;//{0, 1, -0.5, 2.5382};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["cos"] = 1;
    operatorDepth["*"] = 1;
    operatorDepth["+"] = 1;
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>{BinOp::plus(),
    //                                                 Expression<double>{UnOp::cos(), Var{"x3", x3}},
    //                                                 Expression<double>{BinOp::times(), Var{"x0", x0}, Var{"x0", x0}}});

    SymbolicRegressor sr{std::vector<Var>{Var("x0", x0), Var("x1", x1), Var("x2", x2), Var("x3", x3), Var("x4", x4)},
                         std::vector<UnOp>{UnOp::cos()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         2,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    return BenchmarkResult{"PySR",
                           "2.5382*cos(x3)+x0**2-0.5",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_GPLearn(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x0(n);
    x0.setRandom();
    x0 = noiseData(x0, noisePercentage);
    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 = noiseData(x1, noisePercentage);

    auto const y{x0 * x0 - x1 * x1 + x1 - 1};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    std::vector<Expression<double> > extraExpressions;

    SymbolicRegressor sr{std::vector<Var>{Var("x0", x0), Var("x1", x1)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    return BenchmarkResult{"GPLearn",
                           "x0**2-x1**2+x1-1",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_x1Mulx2(double noisePercentage)
{
    srand(0);
    //srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 = noiseData(x1, noisePercentage);
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 = noiseData(x2, noisePercentage);

    auto const y{x1 * x2};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times()},
                         1};

    return BenchmarkResult{"x1Mulx2",
                           "x1*x2",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen1(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    auto const y{x.pow(3) + x.pow(2) + x};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;//{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(),
                                            BinOp::plus()},
                         4,
                         paramsValue};

    return BenchmarkResult{"Nguyen1",
                           "x+x**2+x**3",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen2(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    auto const y{x.pow(4) + x.pow(3) + x.pow(2) + x};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;//{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(),
                                            BinOp::plus()},
                         4,
                         paramsValue};

    return BenchmarkResult{"Nguyen2",
                           "x+x**2+x**3+x**4",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen3(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    auto const y{x.pow(5) + x.pow(4) + x.pow(3) + x.pow(2) + x};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;//{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(),
                                            BinOp::plus()},
                         4,
                         paramsValue};

    return BenchmarkResult{"Nguyen3",
                           "x+x**2+x**3+x**4+x**5",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen4(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    auto const y{x.pow(6) + x.pow(5) + x.pow(4) + x.pow(3) + x.pow(2) + x};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;//{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(),
                                            BinOp::plus()},
                         4,
                         paramsValue};

    return BenchmarkResult{"Nguyen4",
                           "x+x**2+x**3+x**4+x**5+x**6",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen5(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::sin(x * x) * Eigen::cos(x) - 1};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-1, 0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;
    operatorDepth["cos"] = 1;
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>(BinOp::times(), Var("x", x), Var("x", x)));
    //extraExpressions.emplace_back(Expression<double>(BinOp::times(),
    //                                                 Expression<double>(UnOp::cos(), Var("x", x)),
    //                                                 Expression<double>(UnOp::sin(), Expression<double>(BinOp::times(),
    //                                                                                                    Var("x", x),
    //                                                                                                    Var("x", x)))));

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sin(), UnOp::cos()},
                         std::vector<BinOp>{BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};
    //sr.exhaustiveLimit = 1e6; //Optimal expression can generate 3^14 combinations
    sr.keepBestLimit = 25;

    return BenchmarkResult{"Nguyen5",
                           "sin(x**2)*cos(x)-1",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen6(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::sin(x) + Eigen::sin(x + x * x)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>(BinOp::plus(),
    //                                                 Expression<double>(UnOp::sin(), Var("x", x)),
    //                                                 Expression<double>(UnOp::sin(), Expression<double>(BinOp::times(),
    //                                                                                                    Var("x", x),
    //                                                                                                    Var("x", x)))));

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};
    sr.keepBestLimit = 25;

    return BenchmarkResult{"Nguyen6",
                           "sin(x)+sin(x+x**2)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen7(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x += 1;
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::log(x + 1) + Eigen::log(x * x + 1)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["log"] = 1;
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>(BinOp::plus(),
    //                                                 Expression<double>(UnOp::log(), Var("x", x)),
    //                                                 Expression<double>(UnOp::log(), Expression<double>(BinOp::times(),
    //                                                                                                    Var("x", x),
    //                                                                                                    Var("x", x)))));

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::log()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};
    sr.keepBestLimit = 25;

    return BenchmarkResult{"Nguyen7",
                           "log(x+1)+log(x**2+1)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen8(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x += 1;
    x = noiseData(x, noisePercentage);

    auto const y{Eigen::sqrt(x)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sqrt()},
                         std::vector<BinOp>{},
                         2,
                         paramsValue};

    return BenchmarkResult{"Nguyen8",
                           "sqrt(x)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen9(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 = noiseData(x1, noisePercentage);
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 = noiseData(x2, noisePercentage);

    auto const y{Eigen::sin(x1) + Eigen::sin(x2 * x2)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>(BinOp::times(), Var("x2", x2), Var("x2", x2)));
    //extraExpressions.emplace_back(Expression<double>(BinOp::plus(),
    //                                                 Expression<double>(UnOp::sin(), Var("x1", x1)),
    //                                                 Expression<double>(UnOp::sin(), Expression<double>(BinOp::times(),
    //                                                                                                    Var("x2", x2),
    //                                                                                                    Var("x2", x2)))));

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};
    sr.keepBestLimit = 1000;

    return BenchmarkResult{"Nguyen9",
                           "sin(x1)+sin(x2**2)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Nguyen10(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 = noiseData(x1, noisePercentage);
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 = noiseData(x2, noisePercentage);

    auto const y{2 * Eigen::sin(x1) * Eigen::cos(x2)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1, 2};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;
    operatorDepth["cos"] = 1;

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2)},
                         std::vector<UnOp>{UnOp::sin(), UnOp::cos()},
                         std::vector<BinOp>{BinOp::times()},
                         2,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 25;

    return BenchmarkResult{"Nguyen10",
                           "2*sin(x1)*cos(x2)",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_Keijzer10(double noisePercentage)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 += 1;
    x1 = noiseData(x1, noisePercentage);
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 += 1;
    x2 = noiseData(x2, noisePercentage);

    auto const y{Eigen::pow(x1, x2)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{0, 1};

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::pow()},
                         2,
                         paramsValue};

    return BenchmarkResult{"Keijzer10",
                           "x1**x2",
                           sr.fit(noiseData(y, noisePercentage))};
}

BenchmarkResult test_d_bacres1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramValues;//{-1, 0, 0.5, 1, 20};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["/"] = 1;
    std::vector<Expression<double> > extraExpressions;
    extraExpressions.emplace_back(Expression<double>{BinOp::plus(),
                                                     Var("x", data.x),
                                                     Expression<double>{BinOp::divide(),
                                                                        Expression<double>{BinOp::times(),
                                                                                           Var("x", data.x),
                                                                                           Var("y", data.y)},
                                                                        Expression<double>{BinOp::times(),
                                                                                           Var("x", data.x),
                                                                                           Var("x", data.x)}}});

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus(), BinOp::divide()},
                         3,
                         paramValues,
                         operatorDepth,
                         extraExpressions};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_bacres1",
                           "20-x-(x*y/(1+0.5*x**2))",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_bacres2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::map<std::string, size_t> operatorDepth;
    operatorDepth["/"] = 1;
    operatorDepth["*"] = 2;
    operatorDepth["+"] = 2;
    std::vector<double> const paramValues;//{-1, 0, 0.5, 1, 20};
    std::vector<Expression<double> > extraExpressions;
    //extraExpressions.emplace_back(Expression<double>{BinOp::divide(),
    //                                                 Expression<double>{BinOp::times(),
    //                                                                    Var("x", data.x),
    //                                                                    Var("y", data.y)},
    //                                                 Expression<double>{BinOp::times(),
    //                                                                    Var("x", data.x),
    //                                                                    Var("x", data.x)}}});

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus(), BinOp::divide()},
                         3,
                         paramValues,
                         operatorDepth,
                         extraExpressions};
    sr.keepBestLimit = 1000;

    return BenchmarkResult{"d_bacres2",
                           "10-(x*y/(1+0.5*x**2))",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_barmag1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-1, 0, 1, 0.5};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;
    operatorDepth["+"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_barmag1",
                           "0.5*sin(x-y)-sin(x)",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_barmag2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-1, 0, 1, 0.5};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_barmag2",
                           "0.5*sin(y-x)-sin(y)",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_glider1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-1, 0, 1, -0.05};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 50;

    return BenchmarkResult{"d_glider1",
                           "-0.05*x**2-sin(y)",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_glider2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;
/*
    std::vector<double> const paramsValue;//{-1, 0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["cos"] = 1;
    operatorDepth["/"] = 1;
    operatorDepth["+"] = 1;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{UnOp::cos()},
                         std::vector<BinOp>{BinOp::plus(), BinOp::divide()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 50;

    return BenchmarkResult{"d_glider2",
                           "x-cos(y)/x",
                           sr.fit(noiseData(data.label, noisePercentage))};
*/
    auto const cy{data.y.cos()};
    auto const sy{data.y.sin()};

    std::vector<double> const paramsValue;//{-1, 0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["/"] = 1;
    operatorDepth["+"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("cy", cy), Var("sy", sy)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::divide()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 50;

    return BenchmarkResult{"d_glider2",
                           "x-cy/x",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_lv1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_lv1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-3, -2, -1, 0, 1, 2, 3};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["*"] = 3;
    operatorDepth["+"] = 3;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_lv1",
                           "3*x-2*x*y-x**2",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_lv2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_lv2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-2, -1, 0, 1, 2};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["*"] = 3;
    operatorDepth["+"] = 3;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_lv2",
                           "2*y-x*y-y**2",
                           sr.fit(noiseData(data.label, noisePercentage))};
}
/*
BenchmarkResult test_d_predprey1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_predprey1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-1, 0, 1, 4};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["/"] = 1;
    operatorDepth["*"] = 2;
    operatorDepth["+"] = 4;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times(), BinOp::divide()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_predprey1",
                           "x*(4-x-y/(1+x))",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_predprey2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_predprey2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-0.075, 0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["/"] = 1;
    operatorDepth["*"] = 2;
    operatorDepth["+"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times(), BinOp::divide()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_predprey2",
                           "y*(x/(1+x)-0.075*y)",
                           sr.fit(noiseData(data.label, noisePercentage))};
}
*/
BenchmarkResult test_d_shearflow1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_shearflow1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue{-1, 0, 1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["cos"] = 1;
    operatorDepth["cot"] = 1;
    operatorDepth["*"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{UnOp::cos(), UnOp::cot()},
                         std::vector<BinOp>{BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_shearflow1",
                           "cos(x)*cot(y)",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_shearflow2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_shearflow2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;
/*
    std::vector<double> const paramsValue{0, 1, 0.1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["sin"] = 1;
    operatorDepth["cos"] = 1;
    operatorDepth["+"] = 2;
    operatorDepth["*"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{UnOp::cos(), UnOp::sin()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_shearflow2",
                           "(cos(y)**2+0.1*sin(y)**2)*sin(x)",
                           sr.fit(noiseData(data.label, noisePercentage))};
*/
    auto const cx{data.x.cos()};
    auto const sx{data.x.sin()};
    auto const cy{data.y.cos()};
    auto const sy{data.y.sin()};

    std::vector<double> const paramsValue;//{0, 1, 0.1};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["+"] = 2;
    operatorDepth["*"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("cx", cx), Var("sx", sx), Var("cy", cy), Var("sy", sy)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 1000;

    return BenchmarkResult{"d_shearflow2",
                           "(cy**2+0.1*sy**2)*sx",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_vdp1(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_vdp1.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["+"] = 2;
    operatorDepth["*"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_vdp1",
                           "10*(y-(1/3*(x**3-x)))",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

BenchmarkResult test_d_vdp2(double noisePercentage)
{
    auto data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_vdp2.txt")};
    data.x = noiseData(data.x, noisePercentage);
    data.y = noiseData(data.y, noisePercentage);

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["+"] = 2;
    operatorDepth["*"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", data.x), Var("y", data.y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};
    sr.keepBestLimit = 100;

    return BenchmarkResult{"d_vdp2",
                           "-0.1*x",
                           sr.fit(noiseData(data.label, noisePercentage))};
}

void output(BenchmarkResult const& result, std::chrono::milliseconds const& time)
{
    std::cout << "|" << result.name << "|" << result.result.loss << "|" << result.result.time << "|" << time << "|`" << result.expression << "`|`" << result.result.expression.sympyStr(0.1) << "`|\n";
}

int main(int argc, char** argv)
{
    double percentage{0};

    if (argc > 1)
        percentage = std::stod(argv[1]);

    std::cout << "# Benchmark with " << percentage << "% of noise" << std::endl;
    std::cout << "|Test name|MSE|Expression time|Test time|Input symbolic expression|Found symbolic expression|\n";
    std::cout << "|-|-|-|-|-|-|\n";

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_5x1Add7x2Addx3Add8(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_LinearFit(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_LogFit(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Test1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Test2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Test3(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Test4(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Test5(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Test6(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_PySR(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_GPLearn(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_x1Mulx2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen3(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen4(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen5(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen6(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen7(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen8(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen9(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Nguyen10(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_Keijzer10(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_bacres1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_bacres2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_barmag1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_barmag2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_glider1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_glider2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_lv1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_lv2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }
/*
    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_predprey1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_predprey2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }
*/
    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_shearflow1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_shearflow2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_vdp1(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    {
        auto const now{std::chrono::steady_clock::now()};
        auto const result{test_d_vdp2(percentage)};
        output(result, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now));
    }

    return 0;
}
