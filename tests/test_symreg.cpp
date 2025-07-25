#include <gtest/gtest.h>

#include <cmath>
#include <iostream>

#include <boost/algorithm/string.hpp>

#include <curl/curl.h>

#include "SymReg/SymbolicRegressor.h"

//https://github.com/lacava/ode-strogatz

//Commented tests need some work

using namespace sr;

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

TEST(TestSymReg, 5x1Add7x2Addx3Add8)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    Eigen::ArrayXd x3(n);
    x3.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, LinearFit)
{
    //srand(0);
    srand(time(0));

    Eigen::ArrayXd x(7);
    x.setRandom();

    using Var = Variable<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)}};

    auto const y(2 * x + 3);

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, LogFit)
{
    //srand(0);
    srand(time(0));

    Eigen::ArrayXd x(100);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Test1)
{
    //x**2+x+1

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;

    auto const y{x.pow(2) + x + 1};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times()},
                         3};

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Test2)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Test3)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Test4)
{
    //x**2+y**2

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    Eigen::ArrayXd y(n);
    y.setRandom();

    auto const z{x.pow(2) + y.pow(2)};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x), Var("y", y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3};

    auto const r{sr.fit(z)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Test5)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;
    x += 11;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Test6)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x *= 10;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, PySR)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x0(n);
    x0.setRandom();
    Eigen::ArrayXd x1(n);
    x1.setRandom();
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    Eigen::ArrayXd x3(n);
    x3.setRandom();
    Eigen::ArrayXd x4(n);
    x4.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, GPLearn)
{
    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x0(n);
    x0.setRandom();
    Eigen::ArrayXd x1(n);
    x1.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Line)
{
    //srand(0);
    srand(time(0));

    Eigen::Vector2d u{1, 2};
    u /= u.norm();
    Eigen::Vector2d const p0{3, 4};

    size_t constexpr n{10};

    Eigen::ArrayXd x(n);
    Eigen::ArrayXd y(n);

    for (size_t i{0}; i < n; ++i)
    {
        auto const t{10.0 * rand() / RAND_MAX - 5};
        auto const p{t * u + p0};
        x[i] = p[0];
        y[i] = p[1];
    }

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    std::vector<Expression<double> > extraExpressions;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x), Var("y", y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    auto zero{x};
    zero *= 0;

    auto const r{sr.fit(zero)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Circle)
{
    //srand(0);
    srand(time(0));

    Eigen::Array2d const p0{1, 2};
    double const rho{4};

    size_t constexpr n{10};

    Eigen::ArrayXd x(n);
    Eigen::ArrayXd y(n);

    for (size_t i{0}; i < n; ++i)
    {
        auto const theta{2.0 * M_PI * rand() / RAND_MAX};
        x[i] = p0[0] + rho * std::cos(theta);
        y[i] = p0[1] + rho * std::sin(theta);
    }

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    std::vector<Expression<double> > extraExpressions;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x), Var("y", y)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    auto zero{x};
    zero *= 0;

    auto const r{sr.fit(zero)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Plane)
{
    //srand(0);
    srand(time(0));

    Eigen::Vector3d n{1, 2, 3};
    n /= n.norm();
    Eigen::Vector3d u{-4, 5, 6};
    u /= u.norm();
    Eigen::Vector3d v{n.cross(u)};
    v /= v.norm();
    u = v.cross(n);
    Eigen::Vector3d const p0{3, 4, 5};

    size_t constexpr N{10};

    Eigen::ArrayXd x(N);
    Eigen::ArrayXd y(N);
    Eigen::ArrayXd z(N);

    for (size_t i{0}; i < N; ++i)
    {
        auto const t1{10.0 * rand() / RAND_MAX - 5};
        auto const t2{10.0 * rand() / RAND_MAX - 5};
        auto const p{t1 * u + t2 * v + p0};
        x[i] = p[0];
        y[i] = p[1];
        z[i] = p[2];
    }

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    std::vector<Expression<double> > extraExpressions;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x), Var("y", y), Var("z", z)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    auto zero{x};
    zero *= 0;

    auto const r{sr.fit(zero)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Sphere)
{
    //srand(0);
    srand(time(0));

    Eigen::Array3d const p0{1, 2, 3};
    double const rho{4};

    size_t constexpr n{10};

    Eigen::ArrayXd x(n);
    Eigen::ArrayXd y(n);
    Eigen::ArrayXd z(n);

    for (size_t i{0}; i < n; ++i)
    {
        auto const theta{M_PI * rand() / RAND_MAX - M_PI / 2};
        auto const phi{2.0 * M_PI * rand() / RAND_MAX};
        x[i] = p0[0] + rho * std::cos(theta) * std::cos(phi);
        y[i] = p0[1] + rho * std::cos(theta) * std::sin(phi);
        z[i] = p0[2] + rho * std::sin(theta);
    }

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    std::vector<double> const paramsValue;
    std::map<std::string, size_t> operatorDepth;
    std::vector<Expression<double> > extraExpressions;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x), Var("y", y), Var("z", z)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    auto zero{x};
    zero *= 0;

    auto const r{sr.fit(zero)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, x1Mulx2)
{
    srand(0);
    //srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    Eigen::ArrayXd x2(n);
    x2.setRandom();

    auto const y{x1 * x2};

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times()},
                         1};

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen1)
{
    //f(x) = x**3 + x**2 + x

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen2)
{
    //f(x) = x**4 + x**3 + x**2 + x

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen3)
{
    //f(x) = x**5 + x**4 + x**3 + x**2 + x

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen4)
{
    //f(x) = x**6 + x**5 + x**4 + x**3 + x**2 + x

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen5)
{
    //f(x) = sin(x**2)cos(x)-1

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen6)
{
    //f(x) = sin(x)+sin(x+x**2)

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen7)
{
    //f(x) = log(x + 1) + log(x**2 + 1)

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x += 1;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen8)
{
    //f(x) = sqrt(x)

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x(n);
    x.setRandom();
    x += 1;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen9)
{
    //f(x) = sin(x1)+sin(x2**2)

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    Eigen::ArrayXd x2(n);
    x2.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Nguyen10)
{
    //f(x) = 2sin(x1)cos(x2)

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    Eigen::ArrayXd x2(n);
    x2.setRandom();

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, Keijzer10)
{
    //f(x) = x1**x2

    //srand(0);
    srand(time(0));

    size_t constexpr n{100};

    Eigen::ArrayXd x1(n);
    x1.setRandom();
    x1 += 1;
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    x2 += 1;

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

    auto const r{sr.fit(y)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_bacres1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_bacres2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_bacres2.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_barmag1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_barmag2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_barmag2.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_glider1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_glider2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_glider2.txt")};

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


    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_lv1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_lv1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_lv2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_lv2.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}
/*
TEST(TestSymReg, d_predprey1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_predprey1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}
*//*
TEST(TestSymReg, d_predprey2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_predprey2.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}
*/
TEST(TestSymReg, d_shearflow1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_shearflow1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_shearflow2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_shearflow2.txt")};

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


    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_vdp1)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_vdp1.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}

TEST(TestSymReg, d_vdp2)
{
    auto const data{downloadUrlData("https://raw.githubusercontent.com/lacava/ode-strogatz/master/d_vdp2.txt")};

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

    auto const r{sr.fit(data.label)};

    std::cout << r.loss << std::endl;
    std::cout << expr(r.expression.str()) << std::endl;
    std::cout << r.expression.optStr() << std::endl;
    std::cout << r.expression.sympyStr(sr.eps) << std::endl;

    EXPECT_TRUE(r.loss < sr.epsLoss);
}
