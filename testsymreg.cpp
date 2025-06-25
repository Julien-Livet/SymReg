#include <QTest>

class TestSymReg : public QObject
{
    Q_OBJECT

    private slots:
        void testBinaryOperator();
        void testLinearFit();
        void testUnaryOperator();
};

#include <iostream>

#include "SymbolicRegressor.h"

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

QTEST_MAIN(TestSymReg)
#include "testsymreg.moc"
