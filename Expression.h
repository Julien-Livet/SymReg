#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <any>
#include <chrono> //TODO: to remove
#include <string>
#include <vector>

#include <ceres/ceres.h>

#include "BinaryOperator.h"
#include "UnaryOperator.h"
#include "Variable.h"

template <typename T>
std::vector<std::vector<T> > generateCombinations(std::vector<T> const& values, size_t n)
{
    if (values.empty() || n <= 0)
        return std::vector<std::vector<T> >{};

    std::vector<size_t> indices(n, 0);
    std::vector<std::vector<T> > combinations;

    while (true)
    {
        std::vector<T> combination;

        for (size_t i{0}; i < n; ++i)
            combination.emplace_back(values[indices[i]]);

        combinations.emplace_back(combination);

        int pos{n - 1};

        while (pos >= 0)
        {
            if (indices[pos] + 1 < values.size())
            {
                ++indices[pos];

                for (size_t j = pos + 1; j < n; ++j)
                    indices[j] = 0;

                break;
            }
            else
                --pos;
        }

        if (pos < 0)
            break;
    }

    return combinations;
}

bool isSymbol(std::string const& s)
{
    return s == "+" || s == "-" || s == "*" || s == "/";
}

template <typename T>
class Residual;

template <typename T>
class Expression
{
    public:
        T a{1};
        T b{0};

        ceres::Jet<T, 4> ja;
        ceres::Jet<T, 4> jb;

        Expression(Variable<T> const& variable) : operand1_{variable}, operand2_{0}, op_{0}
        {
        }
        
        Expression(UnaryOperator<T> const& op, Expression const& operand) : operand1_{operand}, operand2_{0}, op_{op}
        {
            for (auto const& v: operand.opTree_)
                opTree_.emplace_back(v);

            opTree_.emplace_back(op.name());
        }
        
        Expression(BinaryOperator<T> const& op, Expression const& operand1, Expression const& operand2)
            : operand1_{operand1}, operand2_{operand2}, op_{op}
        {
            for (auto const& v: operand1.opTree_)
                opTree_.emplace_back(v);

            for (auto const& v: operand2.opTree_)
                opTree_.emplace_back(v);

            opTree_.emplace_back(op.name());
        }

        Eigen::Array<T, Eigen::Dynamic, 1> eval() const
        {
            if (op_.type() == typeid(int))
                return a * std::any_cast<Variable<T> >(operand1_).value() + b;
            else if (op_.type() == typeid(UnaryOperator<T>))
                return a * std::any_cast<UnaryOperator<T> >(op_).op()(std::any_cast<Expression>(operand1_).eval()) + b;
            else// if (op_.type() == typeid(BinaryOperator<T>))
                return a * std::any_cast<BinaryOperator<T> >(op_).op()(std::any_cast<Expression>(operand1_).eval(), std::any_cast<Expression>(operand2_).eval()) + b;
        }

        Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1> evalJets() const
        {
            if (op_.type() == typeid(int))
                return ja * std::any_cast<Variable<T> >(operand1_).value() + jb;
            else if (op_.type() == typeid(UnaryOperator<T>))
                return ja * std::any_cast<UnaryOperator<T> >(op_).op()(std::any_cast<Expression>(operand1_).eval()) + jb;
            else// if (op_.type() == typeid(BinaryOperator<T>))
                return ja * std::any_cast<BinaryOperator<T> >(op_).op()(std::any_cast<Expression>(operand1_).eval(), std::any_cast<Expression>(operand2_).eval()) + jb;
        }

        std::string str() const
        {
            std::string s("a*(");
            
            if (op_.type() == typeid(int))
                s += std::any_cast<Variable<T> >(operand1_).name();
            else if (op_.type() == typeid(UnaryOperator<T>))
                s += std::any_cast<UnaryOperator<T> >(op_).name() + "(" + std::any_cast<Expression>(operand1_).str() + ")";
            else// if (op_.type() == typeid(BinaryOperator<T>))
            {
                auto const n{std::any_cast<BinaryOperator<T> >(op_).name()};

                if (isSymbol(n))
                    s += "(" + std::any_cast<Expression>(operand1_).str() + ")" + n + "(" + std::any_cast<Expression>(operand2_).str() + ")";
                else
                    s += n + "(" + std::any_cast<Expression>(operand1_).str() + "," + std::any_cast<Expression>(operand2_).str() + ")";
            }
                
            s += ")+b";
            
            return s;
        }

        std::string opt_str() const
        {
            std::string s;
            
            if (std::abs(a) > std::numeric_limits<T>::epsilon())
            {
                s += std::to_string(a) + "*(";
            
                if (op_.type() == typeid(int))
                    s += std::any_cast<Variable<T> >(operand1_).name();
                else if (op_.type() == typeid(UnaryOperator<T>))
                    s += std::any_cast<UnaryOperator<T> >(op_).name() + "(" + std::any_cast<Expression>(operand1_).opt_str() + ")";
                else// if (op_.type() == typeid(BinaryOperator<T>))
                {
                    auto const n{std::any_cast<BinaryOperator<T> >(op_).name()};

                    if (isSymbol(n))
                        s += "(" + std::any_cast<Expression>(operand1_).opt_str() + ")" + n + "(" + std::any_cast<Expression>(operand2_).opt_str() + ")";
                    else
                        s += n + "(" + std::any_cast<Expression>(operand1_).opt_str() + "," + std::any_cast<Expression>(operand2_).opt_str() + ")";
                }

                s += ")";
            }

            if (std::abs(b) > std::numeric_limits<T>::epsilon())
            {
                if (s.size())
                    s += "+";
            
                s += std::to_string(b);
            }

            if (s.empty())
                s = "0";
            
            return s;
        }

        std::any const& operand1() const
        {
            return operand1_;
        }

        std::any& operand1()
        {
            return operand1_;
        }

        std::any const& operand2() const
        {
            return operand2_;
        }

        std::any& operand2()
        {
            return operand2_;
        }

        std::any const& op() const
        {
            return op_;
        }

        std::any& op()
        {
            return op_;
        }

        void params(std::vector<T>& params) const
        {
            params.emplace_back(a);
            params.emplace_back(b);

            if (operand1_.type() == typeid(Expression))
                std::any_cast<Expression>(operand1_).params(params);

            if (operand2_.type() == typeid(Expression))
                std::any_cast<Expression>(operand2_).params(params);
        }

        template <typename Container>
        void jets(Container& params) const
        {
            params.emplace_back(ja);
            params.emplace_back(jb);

            if (operand1_.type() == typeid(Expression))
                std::any_cast<Expression>(operand1_).jets(params);

            if (operand2_.type() == typeid(Expression))
                std::any_cast<Expression>(operand2_).jets(params);
        }

        std::vector<T> applyParams(std::vector<T> const& params)
        {
            a = params[0];
            b = params[1];

            std::vector<T> p{params.begin() + 2, params.end()};

            if (operand1_.type() == typeid(Expression))
            {
                auto e{std::any_cast<Expression>(operand1_)};
                p = e.applyParams(p);
                operand1_ = e;
            }

            if (operand2_.type() == typeid(Expression))
            {
                auto e{std::any_cast<Expression>(operand2_)};
                p = e.applyParams(p);
                operand2_ = e;
            }

            return p;
        }

        template <typename Container>
        Container applyJets(Container const& params)
        {
            ja = params[0];
            jb = params[1];

            Container p{params.begin() + 2, params.end()};

            if (operand1_.type() == typeid(Expression))
            {
                auto e{std::any_cast<Expression>(operand1_)};
                p = e.applyJets(p);
                operand1_ = e;
            }

            if (operand2_.type() == typeid(Expression))
            {
                auto e{std::any_cast<Expression>(operand2_)};
                p = e.applyJets(p);
                operand2_ = e;
            }

            return p;
        }

        T fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y, std::vector<T> const& paramValues = std::vector<T>{})
        {
            std::vector<double> params;
            this->params(params);

            if (paramValues.size())
            {/*
                auto const t{std::chrono::high_resolution_clock::now()};
                auto const combinations{generateCombinations(paramValues, params.size())};
                std::cout << (std::chrono::high_resolution_clock::now() - t).count() << std::endl;
                T bestCost{std::numeric_limits<T>::infinity()};

                for (auto const& v : combinations)
                {
                    applyParams(v);

                    auto const x{eval()};
                    auto const cost{(y - x).square().sum()};

                    if (cost < bestCost)
                    {
                        bestCost = cost;
                        params = v;
                    }
                }*/
                params = optimizeParamsParallel(paramValues, params.size(), *this, y);

                applyParams(params);
            }

            auto const n{params.size()};

            std::vector<double*> param_ptrs(n);
            for (size_t i = 0; i < n; ++i)
                param_ptrs[i] = &params[i];

            ceres::Problem problem;
            auto* cost = new ceres::DynamicAutoDiffCostFunction<Residual<T> >(
                new Residual<T>(*this, y));

            for (size_t i{0}; i < n; ++i)
                cost->AddParameterBlock(1);

            cost->SetNumResiduals(y.size());

            problem.AddResidualBlock(cost, nullptr, param_ptrs);

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            applyParams(params);

            return summary.final_cost;
        }

        std::vector<std::string> const& opTree() const
        {
            return opTree_;
        }

    private:
        std::any operand1_; //Variable<T> or Expression
        std::any operand2_; //0 or valid operand
        std::any op_; //0 or valid operator
        std::vector<std::string> opTree_;
};

template <typename S>
struct UnderlyingScalar {
    using type = S;
};

template <typename T, int N>
struct UnderlyingScalar<ceres::Jet<T, N>> {
    using type = T;
};

template <typename T>
struct Residual
{
    Residual(Expression<T> const& expression,
             Eigen::Array<double, Eigen::Dynamic, 1> const& y)
        : expression_{expression}, y_{y}
    {
    }

    template <typename S>
    bool operator()(const S* const* p, S* residual)
    {
        using Scalar = typename UnderlyingScalar<S>::type;

        if constexpr (typeid(S) == typeid(T))
        {
            std::vector<Scalar> params;
            expression_.params(params);

            for (size_t i{0}; i < params.size(); ++i)
                params[i] = *p[i];

            expression_.applyParams(params);

            auto const x{expression_.eval()};

            for (int i{0}; i < x.size(); ++i)
                residual[i] = S(y_[i]) - S(x[i]);
        }
        else
        {
            std::vector<S> jets;
            expression_.jets(jets);

            for (size_t i{0}; i < jets.size(); ++i)
                jets[i] = *p[i];

            expression_.applyJets(jets);

            auto const x{expression_.evalJets()};

            for (int i{0}; i < x.size(); ++i)
                residual[i] = S(y_[i]) - S(x[i]);
        }

        return true;
    }

    private:
        Expression<T> expression_;
        const Eigen::Array<double, Eigen::Dynamic, 1> y_;
};

template <typename T, typename S>
void generateAndEvaluate(
    const std::vector<T>& values,
    size_t n,
    std::vector<T>& currentCombination,
    size_t pos,
    T& bestCost,
    std::vector<T>& bestParams,
    Expression<T> expression,
    const S& y)
{
    if (pos == n)
    {
        expression.applyParams(currentCombination);
        auto const x{expression.eval()};
        T cost = (y - x).square().sum();

        if (cost < bestCost)
        {
            bestCost = cost;
            bestParams = currentCombination;
        }

        return;
    }

    for (const auto& val : values)
    {
        currentCombination[pos] = val;
        generateAndEvaluate(values, n, currentCombination, pos + 1, bestCost, bestParams, expression, y);
    }
}

template <typename T, typename S>
std::vector<T> optimizeParamsParallel(
    const std::vector<T>& paramValues,
    size_t n,
    Expression<T> const& expression,
    const S& y)
{
    T globalBestCost = std::numeric_limits<T>::infinity();
    std::vector<T> globalBestParams(n);

    #pragma omp parallel
    {
        T localBestCost = std::numeric_limits<T>::infinity();
        std::vector<T> localBestParams(n);
        std::vector<T> currentCombination(n);

        #pragma omp for nowait
        for (size_t i = 0; i < paramValues.size(); ++i) {
            currentCombination[0] = paramValues[i];
            generateAndEvaluate(paramValues, n, currentCombination, 1, localBestCost, localBestParams, expression, y);
        }

        #pragma omp critical
        {
            if (localBestCost < globalBestCost)
            {
                globalBestCost = localBestCost;
                globalBestParams = localBestParams;
            }
        }
    }

    return globalBestParams;
}

#endif // EXPRESSION_H
