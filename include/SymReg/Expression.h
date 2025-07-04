#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <algorithm>
#include <any>
#include <deque>
#include <random>
#include <string>
#include <vector>

#include <ceres/ceres.h>

#define ARMA_DONT_PRINT_FAST_MATH_WARNING
#include <mlpack/methods/kmeans/kmeans.hpp>

#include "SymReg/BinaryOperator.h"
#include "SymReg/UnaryOperator.h"
#include "SymReg/Variable.h"

namespace sr
{
    bool isSymbol(std::string const& s)
    {
        return s == "+" || s == "-" || s == "*" || s == "/";
    }

    template <typename T>
    std::vector<std::vector<T > > discreteValues(size_t n, size_t m, std::vector<T> const& paramValues)
    {
        std::vector<std::vector<T> > values;
        values.reserve(n);

        for (size_t i{0}; i < n; ++i)
        {
            std::vector<T> v;

            std::random_device rd;
            std::uniform_int_distribution<> d(0, paramValues.size());

            for (size_t j{0}; j < m; ++j)
                v.emplace_back(paramValues[d(rd)]);

            values.emplace_back(v);
        }

        return values;
    }

    template <typename T>
    std::vector<T> roundParams(std::vector<T> const& params, std::vector<T> const& paramValues)
    {
        if (paramValues.empty())
            return params;

        std::vector<T> rounded;
        
        rounded.reserve(params.size());

        for (auto const& p : params)
        {
            auto bestVal = paramValues[0];
            auto bestDist = std::abs(p - bestVal);

            for (const T& v : paramValues)
            {
                auto const dist = std::abs(p - v);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestVal = v;
                }
            }

            rounded.emplace_back(bestVal);
        }

        return rounded;
    }

    template <typename T>
    struct ScoredVector
    {
        std::vector<T> values;
        T distance;

        bool operator<(const ScoredVector<T>& other) const
        {
            return distance < other.distance;
        }
    };

    template <typename T>
    void findBestCombinations(
        std::vector<T> const& paramValues,
        size_t n,
        std::vector<T> const& target,
        size_t maxResults,
        std::vector<std::vector<T> >& bestCombinations)
    {
        std::deque<ScoredVector<T> > topResults;

        std::vector<T> current(n);

        std::function<void(std::size_t)> recurse = [&] (size_t index)
        {
            if (index == n)
            {
                std::vector<T> diff(n);

                for (size_t i = 0; i < n; ++i)
                    diff[i] = current[i] - target[i];

                auto const dist = boost::math::tools::l2_norm(diff);

                auto const it = std::upper_bound(topResults.begin(), topResults.end(), ScoredVector<T>{current, dist});
                topResults.insert(it, ScoredVector<T>{current, dist});

                if (topResults.size() > maxResults)
                    topResults.pop_back();

                return;
            }

            for (auto const& val : paramValues)
            {
                current[index] = val;
                recurse(index + 1);
            }
        };

        recurse(0);

        bestCombinations.clear();

        for (const auto& entry : topResults)
            bestCombinations.push_back(entry.values);
    }

    std::string expr(std::string const& s)
    {
        auto const ca{std::ranges::count(s, 'a')};
        auto const cb{std::ranges::count(s, 'b')};
        auto e{s};
        size_t pos_a{0};
        size_t pos_b{0};

        for (int i{0}; i < ca; ++i)
        {
            pos_a = e.find('a', pos_a);
            e.replace(pos_a, 1, std::string("a") + std::to_string(i));
            pos_a += 1;
        }

        for (int i{0}; i < cb; ++i)
        {
            pos_b = e.find('b', pos_b);
            e.replace(pos_b, 1, std::string("b") + std::to_string(i));
            pos_b += 1;
        }

        return e;
    }

    template <typename T>
    class Residual;

    template <typename T>
    class Expression
    {
        public:
            T a{1};
            T b{1};

            ceres::Jet<T, 4> ja;
            ceres::Jet<T, 4> jb;

            bool aFixed{false};
            bool bFixed{false};

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

                if (op.name() == "+" || op.name() == "-")
                {
                    aFixed = true;

                    auto op1{operand1};
                    op1.bFixed = true;
                    operand1_ = op1;

                    auto op2{operand2};
                    op2.bFixed = true;
                    operand2_ = op2;

                    if (operand1 == operand2)
                    {
                        op_ = 0;
                        operand2_ = 0;
                    }
                }
                else if (op.name() == "*")
                {
                    auto op1_{operand1};
                    op1_.bFixed = true;

                    auto op2_{operand2};
                    op2_.bFixed = true;

                    auto op1{operand1};
                    op1.aFixed = true;
                    op1.bFixed = true;
                    operand1_ = op1;

                    auto op2{operand2};
                    op2.aFixed = true;
                    op2.bFixed = true;
                    operand2_ = op2;

                    bFixed = true;

                    Expression<T> const add{BinaryOperator<T>::plus(), *this,
                                            Expression<T>{BinaryOperator<T>::plus(), op1_, op2_}};

                    *this = std::move(add);
                }
            }

            bool operator==(Expression<T> const& other) const
            {
                if (op_.type() != other.op_.type())
                    return false;

                if (op_.type() == typeid(BinaryOperator<T>) && std::any_cast<BinaryOperator<T> >(op_).symmetry == BinaryOperator<T>::NonStrictSymmetry)
                {
                    if (!(operand1_.type() == other.operand2_.type() && operand2_.type() == other.operand1_.type()))
                        return false;

                    if (!(std::any_cast<Expression<T> >(operand1_) == std::any_cast<Expression<T> >(other.operand2_)
                          && std::any_cast<Expression<T> >(operand2_) == std::any_cast<Expression<T> >(other.operand1_)))
                        return false;
                }

                if (operand1_.type() != other.operand1_.type())
                    return false;

                if (operand1_.type() == typeid(Variable<T>))
                {
                    if (std::any_cast<Variable<T> >(operand1_) != std::any_cast<Variable<T> >(other.operand1_))
                        return false;
                }
                else if (operand1_.type() == typeid(Expression<T>))
                {
                    if (std::any_cast<Expression<T> >(operand1_) != std::any_cast<Expression<T> >(other.operand1_))
                        return false;
                }

                if (operand2_.type() != other.operand2_.type())
                    return false;

                if (operand2_.type() == typeid(Variable<T>))
                {
                    if (std::any_cast<Variable<T> >(operand2_) != std::any_cast<Variable<T> >(other.operand2_))
                        return false;
                }
                else if (operand1_.type() == typeid(Expression<T>))
                {
                    if (std::any_cast<Expression<T> >(operand2_) != std::any_cast<Expression<T> >(other.operand2_))
                        return false;
                }

                return true;
            }

            Eigen::Array<T, Eigen::Dynamic, 1> eval() const
            {
                auto a_{a};
                auto b_{b};

                if (aFixed)
                    a_ = 1;

                if (bFixed)
                    b_ = 0;

                if (op_.type() == typeid(int))
                {
                    if (operand1_.type() == typeid(Variable<T>))
                        return a_ * std::any_cast<Variable<T> >(operand1_).value() + b_;
                    else
                        return a_ * std::any_cast<Expression<T> >(operand1_).eval() + b_;
                }
                else if (op_.type() == typeid(UnaryOperator<T>))
                    return a_ * std::any_cast<UnaryOperator<T> >(op_).op()(std::any_cast<Expression>(operand1_).eval()) + b_;
                else// if (op_.type() == typeid(BinaryOperator<T>))
                    return a_ * std::any_cast<BinaryOperator<T> >(op_).op()(std::any_cast<Expression>(operand1_).eval(), std::any_cast<Expression>(operand2_).eval()) + b_;
            }

            Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1> evalJets() const
            {
                auto ja_{ja};
                auto jb_{jb};

                if (aFixed)
                    ja_ = ceres::Jet<T, 4>{1};

                if (bFixed)
                    jb_ = ceres::Jet<T, 4>{0};

                if (op_.type() == typeid(int))
                {
                    if (operand1_.type() == typeid(Variable<T>))
                        return ja_ * std::any_cast<Variable<T> >(operand1_).value() + jb_;
                    else
                        return ja_ * std::any_cast<Expression<T> >(operand1_).evalJets() + jb_;
                }
                else if (op_.type() == typeid(UnaryOperator<T>))
                    return ja_ * std::any_cast<UnaryOperator<T> >(op_).jetOp()(std::any_cast<Expression>(operand1_).evalJets()) + jb_;
                else// if (op_.type() == typeid(BinaryOperator<T>))
                    return ja_ * std::any_cast<BinaryOperator<T> >(op_).jetOp()(std::any_cast<Expression>(operand1_).evalJets(), std::any_cast<Expression>(operand2_).evalJets()) + jb_;
            }

            std::string str() const
            {
                std::string s;

                if (!aFixed)
                    s = "a*(";
                
                if (op_.type() == typeid(int))
                {
                    if (operand1_.type() == typeid(Variable<T>))
                        s += std::any_cast<Variable<T> >(operand1_).name();
                    else
                        s += std::any_cast<Expression<T> >(operand1_).str();
                }
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

                if (!aFixed)
                    s += ")";

                if (!bFixed)
                    s += "+b";

                return s;
            }

            std::string optStr() const
            {
                auto a_{a};
                auto b_{b};

                if (aFixed)
                    a_ = 1;

                if (bFixed)
                    b_ = 0;

                std::string s;

                if (std::abs(a_) > std::numeric_limits<T>::epsilon())
                {
                    if (!aFixed)
                        s += std::to_string(a_) + "*(";

                    if (op_.type() == typeid(int))
                    {
                        if (operand1_.type() == typeid(Variable<T>))
                            s += std::any_cast<Variable<T> >(operand1_).name();
                        else
                            s += std::any_cast<Expression<T> >(operand1_).optStr();
                    }
                    else if (op_.type() == typeid(UnaryOperator<T>))
                        s += std::any_cast<UnaryOperator<T> >(op_).name() + "(" + std::any_cast<Expression>(operand1_).optStr() + ")";
                    else// if (op_.type() == typeid(BinaryOperator<T>))
                    {
                        auto const n{std::any_cast<BinaryOperator<T> >(op_).name()};

                        if (isSymbol(n))
                            s += "(" + std::any_cast<Expression>(operand1_).optStr() + ")" + n + "(" + std::any_cast<Expression>(operand2_).optStr() + ")";
                        else
                            s += n + "(" + std::any_cast<Expression>(operand1_).optStr() + "," + std::any_cast<Expression>(operand2_).optStr() + ")";
                    }

                    if (!aFixed)
                        s += ")";
                }

                if (std::abs(b_) > std::numeric_limits<T>::epsilon())
                {
                    if (!bFixed)
                    {
                        if (s.size())
                            s += "+";

                        s += std::to_string(b_);
                    }
                }

                if (s.empty())
                    s = "0";

                return s;
            }

            void params(std::vector<T>& params) const
            {
                if (!aFixed)
                    params.emplace_back(a);
                if (!bFixed)
                    params.emplace_back(b);

                if (operand1_.type() == typeid(Expression<T>))
                    std::any_cast<Expression>(operand1_).params(params);

                if (operand2_.type() == typeid(Expression<T>))
                    std::any_cast<Expression>(operand2_).params(params);
            }

            template <typename Container>
            void jets(Container& params) const
            {
                if (!aFixed)
                    params.emplace_back(ja);
                if (!bFixed)
                    params.emplace_back(jb);

                if (operand1_.type() == typeid(Expression<T>))
                    std::any_cast<Expression>(operand1_).jets(params);

                if (operand2_.type() == typeid(Expression<T>))
                    std::any_cast<Expression>(operand2_).jets(params);
            }

            std::vector<T> applyParams(std::vector<T> const& params)
            {
                size_t shift{0};

                if (!aFixed)
                {
                    a = params[shift];
                    ++shift;
                }

                if (!bFixed)
                {
                    b = params[shift];
                    ++shift;
                }

                std::vector<T> p{params.begin() + shift, params.end()};

                if (operand1_.type() == typeid(Expression<T>))
                {
                    auto e{std::any_cast<Expression>(operand1_)};
                    p = e.applyParams(p);
                    operand1_ = e;
                }

                if (operand2_.type() == typeid(Expression<T>))
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
                size_t shift{0};

                if (!aFixed)
                {
                    ja = params[shift];
                    ++shift;
                }

                if (!bFixed)
                {
                    jb = params[shift];
                    ++shift;
                }

                Container p{params.begin() + shift, params.end()};

                if (operand1_.type() == typeid(Expression<T>))
                {
                    auto e{std::any_cast<Expression>(operand1_)};
                    p = e.applyJets(p);
                    operand1_ = e;
                }

                if (operand2_.type() == typeid(Expression<T>))
                {
                    auto e{std::any_cast<Expression>(operand2_)};
                    p = e.applyJets(p);
                    operand2_ = e;
                }

                return p;
            }

            T fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y, std::vector<T> const& paramValues = std::vector<T>{}, T epsLoss = 1e-6, bool verbose = false, size_t exhaustiveLimit = 1e5, bool discreteParams = true)
            {
                std::vector<double> params;
                this->params(params);
                auto const possibilities{std::pow(paramValues.size(), params.size())};
                
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
                options.minimizer_progress_to_stdout = verbose;
                options.logging_type = ceres::SILENT;

                if (paramValues.size())
                {
                    size_t const count{1000};
                    size_t const K{20};
                
                    auto const v{discreteValues(count, params.size(), paramValues)};
                    arma::mat data(count, params.size());

                    for (size_t i{0}; i < count; ++i)
                    {
                        for (size_t j{0}; j < params.size(); ++j)
                            data(i, j) = v[i][j];
                    }

                    arma::Row<size_t> assignments;
                    arma::mat centroids;

                    mlpack::KMeans<> kmeans;
                    kmeans.Cluster(data, K, assignments, centroids);
                    
                    std::vector<T> bestParams(params);
                    T bestCost{std::numeric_limits<T>::infinity()};
                    
                    for (size_t i{0}; i < count; ++i)
                    {
                        for (size_t j{0}; j < params.size(); ++j)
                            params[j] = data(i, j);
                            
                        applyParams(params);
                        
                        ceres::Solver::Summary summary;
                        ceres::Solve(options, &problem, &summary);
                        
                        if (summary.final_cost < bestCost)
                        {
                            bestCost = summary.final_cost;
                            bestParams = params;
                            
                            if (bestCost < epsLoss)
                                break;
                        }
                    }

                    auto const roundedParams{roundParams(bestParams, paramValues)};

                    applyParams(roundedParams);
                    
                    auto const x{eval()};
                    auto const cost{(y - x).square().sum()};
                    
                    return cost;
                }
/*
                if (paramValues.size() && possibilities < exhaustiveLimit)
                {
                    params = optimizeParamsParallel(paramValues, params.size(), *this, y, epsLoss);

                    applyParams(params);

                    auto const x{eval()};
                    auto const cost{(y - x).square().sum()};

                    if (discreteParams)
                        return cost;
                }
*/
                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                //applyParams(params);
/*
                if (!discreteParams && summary.final_cost < epsLoss)
                    return summary.final_cost;

                if (discreteParams || possibilities >= exhaustiveLimit)
                {
                    auto const roundedParams{roundParams(params, paramValues)};

                    std::vector<std::vector<T> > combinations;

                    findBestCombinations(paramValues, params.size(), roundedParams, exhaustiveLimit, combinations);

                    auto bestParams{roundedParams};
                    applyParams(bestParams);
                    auto x{eval()};
                    auto bestCost{(y - x).square().sum()};

                    #pragma omp for nowait
                    for (size_t i = 0; i < combinations.size(); ++i)
                    {
                        auto const& c{combinations[i]};
                        auto e{*this};
                        e.applyParams(c);
                        x = e.eval();
                        auto const cost{(y - x).square().sum()};

                        if (cost < bestCost)
                        {
                            bestCost = cost;
                            bestParams = c;
                        }

                        if (cost < epsLoss)
                            i = combinations.size();
                    }

                    applyParams(bestParams);

                    return bestCost;
                }
*/
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
    struct is_ceres_jet : std::false_type {};

    template <typename T, int N>
    struct is_ceres_jet<ceres::Jet<T, N>> : std::true_type {};

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

            std::vector<Scalar> params;
            expression_.params(params);

            if constexpr (!is_ceres_jet<S>::value)
            {
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
                    residual[i] = S(y_[i]) - x[i];
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
            auto const cost{(y - x).square().sum()};

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
            for (size_t i = 0; i < paramValues.size(); ++i)
            {
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
}

#endif // EXPRESSION_H
