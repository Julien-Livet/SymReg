#include <QTest>

class TestSymReg : public QObject
{
    Q_OBJECT

    private slots:
        void initTestCase();
        void test5x1Add7x2Addx3Add8();
        void test2();
        void test3();
        void test5();
        void test6();
        void testPySR();
        void testx1Mulx2();
        void testLinearFit();
        void testLogFit();
        void testNguyen5();
        void testNguyen6();
        void testNguyen7();
        void testNguyen8();
        void testNguyen9();
        void testNguyen10();
        void testKeijzer10();
};

#include <cmath>
#include <iostream>

#include "SymbolicRegressor.h"

void TestSymReg::initTestCase()
{
    auto const timeout{30 * 60 * 1000};

    qputenv("QTEST_FUNCTION_TIMEOUT", QString::number(timeout).toUtf8());/**
    //TODO: to remove
    size_t constexpr n{100};
    Eigen::ArrayXd x1(n);
    x1.setRandom();
    Eigen::ArrayXd x2(n);
    x2.setRandom();
    Eigen::ArrayXd x3(n);
    x3.setRandom();

    using Var = Variable<double>;
    using BinOp = BinaryOperator<double>;

    Expression<double> const add{BinOp::plus(), Var{"x1", x1}, Var{"x2", x2}};
    std::cout << add.str() << std::endl;
    Expression<double> const mul{BinOp::times(), Var{"x1", x1}, Var{"x2", x2}};
    std::cout << mul.str() << std::endl;
    Expression<double> const add1{BinOp::plus(), add, Var{"x3", x3}};
    std::cout << add1.str() << std::endl;**/
}

void TestSymReg::test5x1Add7x2Addx3Add8()
{
    std::cout << "Running test5x1Add7x2Addx3Add8" << std::endl;

    srand(0);
    //srand(time(0));

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
    std::vector<double> paramsValue;/*
    for (int i{0}; i < ls.size(); ++i)
        paramsValue.emplace_back(ls[i]);*/

    std::map<std::string, size_t> operatorDepth;
    operatorDepth["+"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2), Var("x3", x3)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth,
                         std::vector<Expression<double> >{},
                         true};

    auto const p{sr.fit(y)};
    std::cout << p.first << std::endl;
    std::cout << p.second.str() << std::endl;
    std::cout << p.second.optStr() << std::endl;

    exit(0);

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::test2()
{
    std::cout << "Running test2" << std::endl;

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

    std::vector<double> const paramsValue{0, 1};
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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::test3()
{
    std::cout << "Running test3" << std::endl;

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

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::times(), BinOp::divide()},
                         2,
                         paramsValue};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::test5()
{
    std::cout << "Running test5" << std::endl;

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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::test6()
{
    std::cout << "Running test6" << std::endl;

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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testPySR()
{
    std::cout << "Running testPySR" << std::endl;
/*
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

    std::vector<double> const paramsValue{0, 1, -0.5, 2.5382};
    std::map<std::string, size_t> operatorDepth;
    operatorDepth["cos"] = 1;

    SymbolicRegressor sr{std::vector<Var>{Var("x0", x0), Var("x1", x1), Var("x2", x2), Var("x3", x3), Var("x4", x4)},
                         std::vector<UnOp>{UnOp::cos()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);*/
}

void TestSymReg::testx1Mulx2()
{
    std::cout << "Running testx1Mulx2" << std::endl;

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

    auto const p{sr.fit(y)};
    std::cout << p.first << std::endl;
    std::cout << p.second.str() << std::endl;
    std::cout << p.second.optStr() << std::endl;

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testLinearFit()
{
    std::cout << "Running testLinearFit" << std::endl;

    //srand(0);
    srand(time(0));

    Eigen::ArrayXd x(7);
    x.setRandom();

    using Var = Variable<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)}};

    auto const y(2 * x + 3);

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);

    std::vector<double> params;
    p.second.params(params);

    QVERIFY(std::abs(params[0] - 2) < 1e-6);
    QVERIFY(std::abs(params[1] - 3) < 1e-6);
}

void TestSymReg::testLogFit()
{
    std::cout << "Running testLogFit" << std::endl;

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
                         std::vector<UnOp>{UnOp::log(), UnOp::exp()},
                         std::vector<BinOp>{},
                         1,
                         paramsValue};

    auto const y(2 * Eigen::log(3 * x + 4) + 5);

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testNguyen5()
{
    std::cout << "Running testNguyen5" << std::endl;

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
    operatorDepth["*"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sin(), UnOp::cos()},
                         std::vector<BinOp>{BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testNguyen6()
{
    std::cout << "Running testNguyen6" << std::endl;

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

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testNguyen7()
{
    std::cout << "Running testNguyen7" << std::endl;

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

    SymbolicRegressor sr{std::vector<Var>{Var("x", x)},
                         std::vector<UnOp>{UnOp::log()},
                         std::vector<BinOp>{BinOp::times(), BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testNguyen8()
{
    std::cout << "Running testNguyen8" << std::endl;

    //f(x) = log(x + 1) + log(x**2 + 1)

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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testNguyen9()
{
    std::cout << "Running testNguyen9" << std::endl;
/**
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
    operatorDepth["*"] = 1;
    operatorDepth["+"] = 1;
    std::vector<Expression<double> > extraExpressions;
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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);**/
}

void TestSymReg::testNguyen10()
{
    std::cout << "Running testNguyen10" << std::endl;

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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

void TestSymReg::testKeijzer10()
{
    std::cout << "Running testKeijzer10" << std::endl;

    //f(x) = 2sin(x1)cos(x2)

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

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

QTEST_MAIN(TestSymReg)
#include "testsymreg.moc"
