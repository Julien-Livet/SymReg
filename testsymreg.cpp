#include <QTest>

class TestSymReg : public QObject
{
    Q_OBJECT

    private slots:
        void initTestCase();
        void test5x1Add7x2Addx3Add8();
        void testx1Mulx2();
        void testBinaryOperator();
        void testLinearFit();
        void testLogFit();
        void testUnaryOperator();
        void testNguyen9();
};

#include <iostream>

#include "SymbolicRegressor.h"

void TestSymReg::initTestCase()
{
    auto const timeout{30 * 60 * 1000};

    qputenv("QTEST_FUNCTION_TIMEOUT", QString::number(timeout).toUtf8());
}

void TestSymReg::test5x1Add7x2Addx3Add8()
{/**
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
    for (int i{0}; i < ls.size(); ++i)
        paramsValue.emplace_back(ls[i]);

    std::map<std::string, size_t> operatorDepth;
    operatorDepth["+"] = 2;

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2), Var("x3", x3)},
                         std::vector<UnOp>{},
                         std::vector<BinOp>{BinOp::plus()},
                         3,
                         paramsValue,
                         operatorDepth};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);

    std::cout << p.second.opt_str() << std::endl;**/
}

void TestSymReg::testx1Mulx2()
{
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
                         std::vector<BinOp>{BinOp::times()}};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);

    std::cout << p.second.opt_str() << std::endl;
}

void TestSymReg::testUnaryOperator()
{
    Eigen::ArrayXd x(7);
    x << 0, 1, 2, 3, 4, 5, 6;

    Variable var("x", x);

    Expression<double> logVar{UnaryOperator<double>::log(), var};
    {
        auto e{std::any_cast<Expression<double> >(logVar.operand1())};
        e.b = 1;
        logVar.operand1() = e;
    }

    {
        std::vector<double> params;
        logVar.params(params);
        params[1 + 2] = 1;
        logVar.applyParams(params);
    }

    auto const y{1 * Eigen::log(1 * x + 1) + 0};
    auto const r{logVar.eval()};

    for (int i{0}; i < y.size(); ++i)
        QVERIFY(std::abs(y[i] - r[i]) < 1e-6);
/*
    std::cout << logVar.str() << std::endl;
    std::cout << logVar.opt_str() << std::endl;
    std::cout << logVar.eval().transpose() << std::endl;*/
}

void TestSymReg::testBinaryOperator()
{
    Eigen::ArrayXd x(7);
    x << 0, 1, 2, 3, 4, 5, 6;

    Variable var("x", x);
    auto var_bis(var);

    Expression<double> addVars{BinaryOperator<double>::plus(), var, var_bis};
    {
        auto e{std::any_cast<Expression<double> >(addVars.operand1())};
        e.b = 1;
        addVars.operand1() = e;
    }

    {
        auto e{std::any_cast<Expression<double> >(addVars.operand2())};
        e.a = 2;
        e.b = 3;
        addVars.operand2() = e;
    }

    {
        std::vector<double> params;
        addVars.params(params);
        params[1 + 2] = 1;
        params[0 + 4] = 2;
        params[1 + 4] = 4;
        addVars.applyParams(params);
    }

    auto const y{1*((1*x+1)+(2*x+4))};
    auto const r{addVars.eval()};

    for (int i{0}; i < y.size(); ++i)
        QVERIFY(std::abs(y[i] - r[i]) < 1e-6);
/*
    std::cout << addVars.str() << std::endl;
    std::cout << addVars.opt_str() << std::endl;
    std::cout << addVars.eval().transpose() << std::endl;*/
}

void TestSymReg::testLinearFit()
{
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

void TestSymReg::testNguyen9()
{
    //f(x) = sin(x1)+sin(x2**2)

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
    std::vector<Expression<double> > extraExpressions;/*
    extraExpressions.emplace_back(Expression<double>(BinOp::plus(),
                                                     Expression<double>(UnOp::sin(), Var("x1", x1)),
                                                     Expression<double>(UnOp::sin(), Expression<double>(BinOp::times(),
                                                                                                        Var("x2", x2),
                                                                                                        Var("x2", x2)))));*/

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1), Var("x2", x2)},
                         std::vector<UnOp>{UnOp::sin()},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                         3,
                         paramsValue,
                         operatorDepth,
                         extraExpressions};

    auto const p{sr.fit(y)};

    QVERIFY(p.first < 1e-6);
}

QTEST_MAIN(TestSymReg)
#include "testsymreg.moc"
