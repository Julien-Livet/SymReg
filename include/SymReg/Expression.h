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

        std::random_device rd;
        std::uniform_int_distribution<> d(0, paramValues.size() - 1);

        for (size_t i{0}; i < n; ++i)
        {
            std::vector<T> v;
            v.reserve(m);

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
            
            enum OperandType
            {
                NoOperand,
                VariableOperand,
                ExpressionOperand
            };
            
            enum OperatorType
            {
                LinearOp,
                UnaryOp,
                BinaryOp
            };

            Expression(Variable<T> const& variable) : operand1Type_{VariableOperand}, operand1Variable_{std::make_unique<Variable<T> >(variable)}, operand1Expression_{},
                                                      operand2Type_{NoOperand}, operand2Variable_{}, operand2Expression_{},
                                                      operatorType_{LinearOp}, unaryOperator_{}, binaryOperator_{}
            {
            }
            
            Expression(UnaryOperator<T> const& op, Expression const& operand)
             : operand1Type_{ExpressionOperand}, operand1Variable_{}, operand1Expression_{std::make_unique<Expression<T> >(operand)},
               operand2Type_{NoOperand}, operand2Variable_{}, operand2Expression_{},
               operatorType_{UnaryOp}, unaryOperator_{std::make_unique<UnaryOperator<T> >(op)}, binaryOperator_{}
            {
                for (auto const& v: operand.opTree_)
                    opTree_.emplace_back(v);

                opTree_.emplace_back(op.name());
            }
            
            Expression(BinaryOperator<T> const& op, Expression const& operand1, Expression const& operand2)
             : operand1Type_{ExpressionOperand}, operand1Variable_{}, operand1Expression_{std::make_unique<Expression<T> >(operand1)},
               operand2Type_{ExpressionOperand}, operand2Variable_{}, operand2Expression_{std::make_unique<Expression<T> >(operand2)},
               operatorType_{BinaryOp}, unaryOperator_{}, binaryOperator_{std::make_unique<BinaryOperator<T> >(op)}
            {
                for (auto const& v: operand1.opTree_)
                    opTree_.emplace_back(v);

                for (auto const& v: operand2.opTree_)
                    opTree_.emplace_back(v);

                opTree_.emplace_back(op.name());

                if (op.name() == "+" || op.name() == "-")
                {
                    aFixed = true;

                    operand1Expression_->bFixed = true;
                    operand2Expression_->bFixed = true;

                    if (*operand1Expression_ == *operand2Expression_)
                    {
                        operatorType_ = LinearOp;
                        operand2Type_ = NoOperand;
                        operand2Expression_.reset();
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
                    operand1Expression_ = std::make_unique<Expression<T> >(op1);

                    auto op2{operand2};
                    op2.aFixed = true;
                    op2.bFixed = true;
                    operand2Expression_ = std::make_unique<Expression<T> >(op2);

                    bFixed = true;

                    Expression<T> const add{BinaryOperator<T>::plus(), *this,
                                            Expression<T>{BinaryOperator<T>::plus(), op1_, op2_}};

                    *this = std::move(add);
                }
            }
            
            Expression(Expression<T> const& other)
             : operand1Type_{other.operand1Type_}, operand1Variable_{}, operand1Expression_{},
               operand2Type_{other.operand2Type_}, operand2Variable_{}, operand2Expression_{},
               operatorType_{other.operatorType_}, unaryOperator_{}, binaryOperator_{},
               aFixed{other.aFixed}, bFixed{other.bFixed}, a{other.a}, b{other.b},
               ja{other.ja}, jb{other.jb}     
            {
                if (other.operand1Variable_)
                    operand1Variable_ = std::make_unique<Variable<T> >(*other.operand1Variable_);

                if (other.operand1Expression_)
                    operand1Expression_ = std::make_unique<Expression<T> >(*other.operand1Expression_);

                if (other.operand2Variable_)
                    operand2Variable_ = std::make_unique<Variable<T> >(*other.operand2Variable_);

                if (other.operand2Expression_)
                    operand2Expression_ = std::make_unique<Expression<T> >(*other.operand2Expression_);

                if (other.unaryOperator_)
                    unaryOperator_ = std::make_unique<UnaryOperator<T> >(*other.unaryOperator_);

                if (other.binaryOperator_)
                    binaryOperator_ = std::make_unique<BinaryOperator<T> >(*other.binaryOperator_);
            }

            Expression& operator=(Expression<T> const& other)
            {
                operand1Type_ = other.operand1Type_;
                operand2Type_ = other.operand2Type_;
                operatorType_ = other.operatorType_;
                aFixed = other.aFixed;
                bFixed = other.bFixed;
                a = other.a;
                b = other.b;
                ja = other.ja;
                jb = other.jb;

                if (other.operand1Variable_)
                    operand1Variable_ = std::make_unique<Variable<T> >(*other.operand1Variable_);

                if (other.operand1Expression_)
                    operand1Expression_ = std::make_unique<Expression<T> >(*other.operand1Expression_);

                if (other.operand2Variable_)
                    operand2Variable_ = std::make_unique<Variable<T> >(*other.operand2Variable_);

                if (other.operand2Expression_)
                    operand2Expression_ = std::make_unique<Expression<T> >(*other.operand2Expression_);

                if (other.unaryOperator_)
                    unaryOperator_ = std::make_unique<UnaryOperator<T> >(*other.unaryOperator_);

                if (other.binaryOperator_)
                    binaryOperator_ = std::make_unique<BinaryOperator<T> >(*other.binaryOperator_);

                return *this;
            }

            bool operator==(Expression<T> const& other) const
            {
                if (operatorType_ != other.operatorType_)
                    return false;

                if (operatorType_ == BinaryOp && binaryOperator_->symmetry == BinaryOperator<T>::NonStrictSymmetry)
                {
                    if (*operand1Expression_ == *other.operand2Expression_ && *operand2Expression_ == *other.operand1Expression_)
                        return true;
                }

                if (operand1Type_ != other.operand1Type_)
                    return false;

                if (operand1Type_ == VariableOperand && *operand1Variable_ != *other.operand1Variable_)
                    return false;

                if (operand1Type_ == ExpressionOperand && *operand1Expression_ != *other.operand1Expression_)
                    return false;

                if (operand2Type_ != other.operand2Type_)
                    return false;

                if (operand2Type_ == VariableOperand && *operand2Variable_ != *other.operand2Variable_)
                    return false;

                if (operand2Type_ == ExpressionOperand && *operand2Expression_ != *other.operand2Expression_)
                    return false;

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

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        return a_ * operand1Variable_->value() + b_;
                    else
                        return a_ * operand1Expression_->eval() + b_;
                }
                else if (operatorType_ == UnaryOp)
                    return a_ * unaryOperator_->op()(operand1Expression_->eval()) + b_;
                else// if (operatorType_ == BinaryOp)
                    return a_ * binaryOperator_->op()(operand1Expression_->eval(), operand2Expression_->eval()) + b_;
            }

            Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1> evalJets() const
            {
                auto ja_{ja};
                auto jb_{jb};

                if (aFixed)
                    ja_ = ceres::Jet<T, 4>{1};

                if (bFixed)
                    jb_ = ceres::Jet<T, 4>{0};

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        return ja_ * operand1Variable_->value() + jb_;
                    else
                        return ja_ * operand1Expression_->evalJets() + jb_;
                }
                else if (operatorType_ == UnaryOp)
                    return ja_ * unaryOperator_->jetOp()(operand1Expression_->evalJets()) + jb_;
                else// if (operatorType_ == BinaryOp)
                    return ja_ * binaryOperator_->jetOp()(operand1Expression_->evalJets(), operand2Expression_->evalJets()) + jb_;
            }

            std::string str() const
            {
                std::string s;

                if (!aFixed)
                    s = "a*(";

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        s += operand1Variable_->name();
                    else
                        s += operand1Expression_->str();
                }
                else if (operatorType_ == UnaryOp)
                    s += unaryOperator_->name() + "(" + operand1Expression_->str() + ")";
                else// if (operatorType_ == BinaryOp)
                {
                    auto const n{binaryOperator_->name()};

                    if (isSymbol(n))
                        s += "(" + operand1Expression_->str() + ")" + n + "(" + operand2Expression_->str() + ")";
                    else
                        s += n + "(" + operand1Expression_->str() + "," + operand2Expression_->str() + ")";
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

                    if (operatorType_ == LinearOp)
                    {
                        if (operand1Type_ == VariableOperand)
                            s += operand1Variable_->name();
                        else
                            s += operand1Expression_->optStr();
                    }
                    else if (operatorType_ == UnaryOp)
                        s += unaryOperator_->name() + "(" + operand1Expression_->optStr() + ")";
                    else// if (operatorType_ == BinaryOp)
                    {
                        auto const n{binaryOperator_->name()};

                        if (isSymbol(n))
                            s += "(" + operand1Expression_->optStr() + ")" + n + "(" + operand2Expression_->optStr() + ")";
                        else
                            s += n + "(" + operand1Expression_->optStr() + "," + operand2Expression_->optStr() + ")";
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

                if (operand1Type_ == ExpressionOperand)
                    operand1Expression_->params(params);

                if (operand2Type_ == ExpressionOperand)
                    operand2Expression_->params(params);
            }

            template <typename Container>
            void jets(Container& params) const
            {
                if (!aFixed)
                    params.emplace_back(ja);

                if (!bFixed)
                    params.emplace_back(jb);

                if (operand1Type_ == ExpressionOperand)
                    operand1Expression_->jets(params);

                if (operand2Type_ == ExpressionOperand)
                    operand2Expression_->jets(params);
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

                if (operand1Type_ == ExpressionOperand)
                    p = operand1Expression_->applyParams(p);

                if (operand2Type_ == ExpressionOperand)
                    p = operand2Expression_->applyParams(p);

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

                if (operand1Type_ == ExpressionOperand)
                    p = operand1Expression_->applyJets(p);

                if (operand2Type_ == ExpressionOperand)
                    p = operand2Expression_->applyJets(p);

                return p;
            }

            T fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y, std::vector<T> const& paramValues = std::vector<T>{}, T epsLoss = 1e-6, bool verbose = false, size_t exhaustiveLimit = 1e5, bool discreteParams = true)
            {
                std::vector<double> params;
                this->params(params);
                auto const possibilities{std::pow(paramValues.size(), params.size())};
                auto const n{params.size()};

                if (paramValues.size() && possibilities < exhaustiveLimit)
                {
                    params = optimizeParamsParallel(paramValues, params.size(), *this, y, epsLoss);

                    applyParams(params);

                    auto const x{eval()};
                    auto const cost{(y - x).square().sum()};

                    if (discreteParams)
                        return cost;
                }

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
                    size_t const count{10000};
                    size_t const K{1000};
                
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
                    
                    for (size_t i{0}; i < K; ++i)
                    {
                        for (size_t j{0}; j < params.size(); ++j)
                            params[j] = centroids(j, i);

                        applyParams(params);
                        
                        ceres::Solver::Summary summary;
                        ceres::Solve(options, &problem, &summary);

                        auto loss{summary.final_cost};

                        if (loss < 0)
                            loss = std::numeric_limits<T>::infinity();

                        if (loss < bestCost)
                        {
                            bestCost = loss;
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

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                auto loss{summary.final_cost};

                if (loss < 0)
                    loss = std::numeric_limits<T>::infinity();

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
                return loss;
            }

            std::vector<std::string> const& opTree() const
            {
                return opTree_;
            }

        private:
            OperandType operand1Type_;
            std::unique_ptr<Variable<T> > operand1Variable_;
            std::unique_ptr<Expression<T> > operand1Expression_;
            OperandType operand2Type_;
            std::unique_ptr<Variable<T> > operand2Variable_;
            std::unique_ptr<Expression<T> > operand2Expression_;
            OperatorType operatorType_;
            std::unique_ptr<UnaryOperator<T> > unaryOperator_;
            std::unique_ptr<BinaryOperator<T> > binaryOperator_;
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
                {
                    if (ceres::isnan(x[i]))
                        return false;

                    residual[i] = S(y_[i]) - S(x[i]);
                }
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
                {
                    if (ceres::isnan(x[i]))
                        return false;

                    residual[i] = S(y_[i]) - x[i];
                }
            }

            return true;
        }

        private:
            Expression<T> expression_;
            const Eigen::Array<double, Eigen::Dynamic, 1> y_;
    };

    template <typename T, typename S>
    bool generateAndEvaluate(
        const std::vector<T>& values,
        size_t n,
        std::vector<T>& currentCombination,
        size_t pos,
        T& bestCost,
        std::vector<T>& bestParams,
        Expression<T> expression,
        const S& y,
        T epsLoss)
    {
        if (pos == n)
        {
            expression.applyParams(currentCombination);
            auto const x{expression.eval()};
            auto cost{(y - x).square().sum()};

            if (std::isnan(cost))
                cost = std::numeric_limits<T>::infinity();

            if (cost < bestCost)
            {
                bestCost = cost;
                bestParams = currentCombination;
            }

            if (cost < epsLoss)
                return true;

            return false;
        }

        for (const auto& val : values)
        {
            currentCombination[pos] = val;

            if (generateAndEvaluate(values, n, currentCombination, pos + 1, bestCost, bestParams, expression, y, epsLoss))
                return true;
        }

        return false;
    }

    template <typename T, typename S>
    std::vector<T> optimizeParamsParallel(
        const std::vector<T>& paramValues,
        size_t n,
        Expression<T> const& expression,
        const S& y,
        T epsLoss)
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

                if (generateAndEvaluate(paramValues, n, currentCombination, 1, localBestCost, localBestParams, expression, y, epsLoss))
                    i = paramValues.size();
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
