#ifndef SYMBOLICREGRESSOR_H
#define SYMBOLICREGRESSOR_H

#include <limits>
#include <map>
#include <vector>

#include <boost/math/tools/norms.hpp>

#include "SymReg/BinaryOperator.h"
#include "SymReg/Expression.h"
#include "SymReg/UnaryOperator.h"
#include "SymReg/Variable.h"

namespace sr
{
    template <typename T>
    class SymbolicRegressor
    {
        public:
            T eps = 1e-4;
            T epsLoss = 1e-12;
            size_t exhaustiveLimit = 1e5;

            SymbolicRegressor(std::vector<Variable<T> > const& variables,
                              std::vector<UnaryOperator<T> > const& un_ops = std::vector<UnaryOperator<T> >{},
                              std::vector<BinaryOperator<T> > const& bin_ops = std::vector<BinaryOperator<T> >{},
                              size_t niterations = 5,
                              std::vector<T> const& paramValues = std::vector<T>{},
                              std::map<std::string, size_t> const& operatorDepth = std::map<std::string, size_t>{},
                              std::vector<Expression<T> > const& extraExpressions = std::vector<Expression<T> >{},
                              bool verbose = false,
                              std::function<void(Expression<T> const&, T const&)> const& callback = [] (Expression<T> const&, T const&) {},
                              bool discreteParams = true)
                : variables_{variables}, un_ops_{un_ops}, bin_ops_{bin_ops},
                  niterations_{niterations}, paramValues_{paramValues},
                  operatorDepth_{operatorDepth}, extraExpressions_{extraExpressions},
                  verbose_{verbose}, callback_(callback), discreteParams_{discreteParams}
            {
                if (!verbose)
                    auto const f{freopen("/tmp/stderr.txt", "w", stderr)};
            }

            std::vector<Variable<T> > const& variables() const
            {
                return variables_;
            }

            std::vector<UnaryOperator<T> > const& un_ops() const
            {
                return un_ops_;
            }

            std::vector<BinaryOperator<T> > const& bin_ops() const
            {
                return bin_ops_;
            }
            
            std::pair<T, Expression<T> > fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y)
            {
                std::vector<Expression<T> > expressions;
                std::vector<T> costs;

                for (auto const& v : variables_)
                {
                    Expression<T> e{v};
                    auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_)};

                    std::vector<T> params;
                    e.params(params);

                    if (boost::math::tools::l2_norm(y) < eps
                        && (e.sympyStr() == "0.0" || e.sympyStr() == "0"))
                        cost = std::numeric_limits<T>::infinity();

                    expressions.emplace_back(e);
                    costs.emplace_back(cost);
                    
                    callback_(e, cost);

                    if (cost < epsLoss)
                        break;
                }

                for (auto const& expression : extraExpressions_)
                {
                    Expression<T> e{expression};
                    auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_)};

                    std::vector<T> params;
                    e.params(params);

                    if (boost::math::tools::l2_norm(y) < eps
                        && (e.sympyStr() == "0.0" || e.sympyStr() == "0"))
                        cost = std::numeric_limits<T>::infinity();

                    expressions.emplace_back(e);
                    costs.emplace_back(cost);
                    
                    callback_(e, cost);

                    if (cost < epsLoss)
                        break;
                }

                std::vector<std::pair<T, Expression<T> > > paired;

                {
                    for (size_t i{0}; i < costs.size(); ++i)
                        paired.emplace_back(costs[i], expressions[i]);

                    std::sort(paired.begin(), paired.end(), [] (auto const& x, auto const& y) {return x.first < y.first;});

                    for (size_t i{0}; i < paired.size(); ++i)
                        expressions[i] = paired[i].second;

                    if (!paired.empty() && paired.front().first < epsLoss)
                        return paired.front();
                }

                std::map<size_t, std::vector<size_t> > unIndices;
                std::map<size_t, std::vector<std::pair<size_t, size_t> > > binIndices;

                for (size_t i{0}; i < un_ops_.size(); ++i)
                    unIndices[i] = std::vector<size_t>{};

                for (size_t i{0}; i < bin_ops_.size(); ++i)
                    binIndices[i] = std::vector<std::pair<size_t, size_t> >{};

                for (size_t i{0}; i < niterations_; ++i)
                {
                    //#pragma omp parallel
                    {
                        size_t const n{expressions.size()};
                        std::vector<Expression<T> > localExpressions;
                        std::vector<double> localCosts;
                        std::map<size_t, std::vector<size_t> > localUnIndices;

                        //#pragma omp for nowait
                        for (size_t j = 0; j < n; ++j)
                        {
                            for (size_t k = 0; k < un_ops_.size(); ++k)
                            {
                                auto const it{std::find(unIndices[k].begin(), unIndices[k].end(), j)};

                                if (it == unIndices[k].end())
                                {
                                    localUnIndices[k].emplace_back(j);

                                    auto const count{std::count(expressions[j].opTree().begin(), expressions[j].opTree().end(), un_ops_[k].name())};
                                    auto maxCount{std::numeric_limits<int>::max()};

                                    if (operatorDepth_.find(un_ops_[k].name()) != operatorDepth_.end())
                                        maxCount = operatorDepth_[un_ops_[k].name()];

                                    if (count < maxCount)
                                    {
                                        Expression<T> e{un_ops_[k], expressions[j]};
                                        auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_)};

                                        std::vector<T> params;
                                        e.params(params);

                                        if (boost::math::tools::l2_norm(y) < eps
                                            && (e.sympyStr() == "0.0" || e.sympyStr() == "0"))
                                            cost = std::numeric_limits<T>::infinity();

                                        localExpressions.emplace_back(e);
                                        localCosts.emplace_back(cost);
                    
                                        callback_(e, cost);

                                        if (cost < epsLoss)
                                        {
                                            j = n;
                                            break;
                                        }
                                    }
                                }
                            }
                        }

                        //#pragma omp critical
                        {
                            expressions.insert(expressions.end(), localExpressions.begin(), localExpressions.end());
                            costs.insert(costs.end(), localCosts.begin(), localCosts.end());

                            for (auto& pair: localUnIndices)
                            {
                                auto& vec = unIndices[pair.first];
                                vec.insert(vec.end(),
                                           std::make_move_iterator(pair.second.begin()),
                                           std::make_move_iterator(pair.second.end()));
                            }
                        }
                    }

                    {
                        paired.clear();

                        for (size_t i{0}; i < costs.size(); ++i)
                            paired.emplace_back(costs[i], expressions[i]);

                        std::sort(paired.begin(), paired.end(), [] (auto const& x, auto const& y) {return x.first < y.first;});

                        for (size_t i{0}; i < paired.size(); ++i)
                            expressions[i] = paired[i].second;

                        if (!paired.empty() && paired.front().first < epsLoss)
                            return paired.front();
                    }

                    //#pragma omp parallel
                    {
                        size_t const n{expressions.size()};
                        std::vector<Expression<T>> localExpressions;
                        std::vector<double> localCosts;
                        std::map<size_t, std::vector<std::pair<size_t, size_t> > > localBinIndices;

                        //#pragma omp for nowait
                        for (size_t j1 = 0; j1 < n; ++j1)
                        {
                            for (size_t k = 0; k < bin_ops_.size(); ++k)
                            {
                                size_t j2{0};

                                if (bin_ops_[k].symmetry == BinaryOperator<T>::NonStrictSymmetry)
                                    j2 = j1;
                                else if (bin_ops_[k].symmetry == BinaryOperator<T>::StrictSymmetry)
                                    j2 = j1 + 1;

                                for (; j2 < n; ++j2)
                                {
                                    auto const it{std::find(binIndices[k].begin(), binIndices[k].end(), std::make_pair(j1, j2))};

                                    if (it == binIndices[k].end())
                                    {
                                        localBinIndices[k].emplace_back(j1, j2);

                                        auto const count1{std::count(expressions[j1].opTree().begin(), expressions[j1].opTree().end(), bin_ops_[k].name())};
                                        auto const count2{std::count(expressions[j2].opTree().begin(), expressions[j2].opTree().end(), bin_ops_[k].name())};
                                        auto maxCount{std::numeric_limits<int>::max()};

                                        if (operatorDepth_.find(bin_ops_[k].name()) != operatorDepth_.end())
                                            maxCount = operatorDepth_[bin_ops_[k].name()];

                                        if (count1 < maxCount && count2 < maxCount)
                                        {
                                            Expression<T> e{bin_ops_[k], expressions[j1], expressions[j2]};
                                            auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_)};

                                            std::vector<T> params;
                                            e.params(params);

                                            if (boost::math::tools::l2_norm(y) < eps
                                                && (e.sympyStr() == "0.0" || e.sympyStr() == "0"))
                                                cost = std::numeric_limits<T>::infinity();

                                            localExpressions.emplace_back(e);
                                            localCosts.emplace_back(cost);

                                            callback_(e, cost);

                                            if (cost < epsLoss)
                                            {
                                                k = bin_ops_.size();
                                                j1 = n;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }

                        //#pragma omp critical
                        {
                            expressions.insert(expressions.end(), localExpressions.begin(), localExpressions.end());
                            costs.insert(costs.end(), localCosts.begin(), localCosts.end());

                            for (auto& pair: localBinIndices)
                            {
                                auto& vec = binIndices[pair.first];
                                vec.insert(vec.end(),
                                           std::make_move_iterator(pair.second.begin()),
                                           std::make_move_iterator(pair.second.end()));
                            }
                        }
                    }

                    {
                        paired.clear();

                        for (size_t i{0}; i < costs.size(); ++i)
                            paired.emplace_back(costs[i], expressions[i]);

                        std::sort(paired.begin(), paired.end(), [] (auto const& x, auto const& y) {return x.first < y.first;});

                        for (size_t i{0}; i < paired.size(); ++i)
                            expressions[i] = paired[i].second;

                        if (!paired.empty() && paired.front().first < epsLoss)
                            return paired.front();
                    }
                }

                if (paired.empty())
                    throw std::runtime_error("No expression found!");

                return paired.front();
            }

        private:
            std::vector<Variable<T> > variables_;
            std::vector<UnaryOperator<T> > un_ops_;
            std::vector<BinaryOperator<T> > bin_ops_;
            size_t niterations_;
            std::vector<T> paramValues_;
            std::map<std::string, size_t> operatorDepth_;
            std::vector<Expression<T> > extraExpressions_;
            bool verbose_;
            std::function<void(Expression<T> const&, T const&)> callback_;
            bool discreteParams_;
    };
}

#endif // SYMBOLICREGRESSOR_H
