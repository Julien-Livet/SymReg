#include <QTest>

class TestSymReg : public QObject
{
    Q_OBJECT

    private slots:
        void test();
};

#include <iostream>

#include "SymbolicRegressor.h"

void TestSymReg::test()
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

    std::cout << logVar.str() << std::endl;
    std::cout << logVar.opt_str() << std::endl;
    std::cout << logVar.eval().transpose() << std::endl;

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

    std::cout << addVars.str() << std::endl;
    std::cout << addVars.opt_str() << std::endl;
    std::cout << addVars.eval().transpose() << std::endl;

    Eigen::ArrayXd x1(7);
    x1.setRandom();
    Eigen::ArrayXd x2(x1.size());
    x2.setRandom();

    using Var = Variable<double>;
    using UnOp = UnaryOperator<double>;
    using BinOp = BinaryOperator<double>;

    SymbolicRegressor sr{std::vector<Var>{Var("x1", x1)/*, Var("x1", x2)*/},
                         std::vector<UnOp>{UnOp::log(), UnOp::exp()},
                         std::vector<BinOp>{BinOp::plus(), BinOp::times()}};

    //auto const y(x1 + x2);
    auto const y(2 * x1 + 3);

    sr.fit(y);
}

QTEST_MAIN(TestSymReg)
#include "testsymreg.moc"
