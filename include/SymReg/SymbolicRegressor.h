#ifndef SYMREG_SYMBOLICREGRESSOR_H
#define SYMREG_SYMBOLICREGRESSOR_H

#include <atomic>
#include <chrono>
#include <future>
#include <limits>
#include <map>
#include <thread>
#include <vector>

#include <boost/asio.hpp>
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
            struct Result
            {
                T loss;
                Expression<T> expression;
                std::chrono::milliseconds time;
            };
        
            T eps = 1e-4;
            T epsLoss = 1e-12;
            size_t exhaustiveLimit = 1e5;
            size_t keepBestLimit;

            SymbolicRegressor(std::vector<Variable<T> > const& variables,
                              std::vector<UnaryOperator<T> > const& un_ops = std::vector<UnaryOperator<T> >{},
                              std::vector<BinaryOperator<T> > const& bin_ops = std::vector<BinaryOperator<T> >{},
                              size_t niterations = 5,
                              std::vector<T> const& paramValues = std::vector<T>{},
                              std::map<std::string, size_t> const& operatorDepth = std::map<std::string, size_t>{},
                              std::vector<Expression<T> > const& extraExpressions = std::vector<Expression<T> >{},
                              bool verbose = false,
                              std::function<void(Result const&)> const& callback = [] (Result const&) {},
                              bool discreteParams = true,
                              size_t timeout = 30 * 60,
                              size_t keepBestLimit = 10000)
                : variables_{variables}, un_ops_{un_ops}, bin_ops_{bin_ops},
                  niterations_{niterations}, paramValues_{paramValues},
                  operatorDepth_{operatorDepth}, extraExpressions_{extraExpressions},
                  verbose_{verbose}, callback_(callback), discreteParams_{discreteParams},
                  timeout_{timeout}, keepBestLimit{keepBestLimit}
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

            Result fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y)
            {
                isTimeout_ = false;

                auto const now{std::chrono::steady_clock::now()};
                size_t keepBestCount{0};
                std::vector<Expression<T> > expressions;

                std::vector<Result> results;

                std::atomic<bool> timeoutTriggered = false;
                boost::asio::io_context io;

                boost::asio::steady_timer timer(io, std::chrono::seconds(timeout_));

                timer.async_wait([&, this] (boost::system::error_code const& ec)
                                 {
                                     if (!ec)
                                     {
                                         isTimeout_ = true;
                                         timeoutTriggered = true;
                                         io.stop();
                                     }
                                 });

                std::thread io_thread([&] () { io.run(); });

                for (auto const& v : variables_)
                {
                    Expression<T> e{v};
                    auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_, timeoutTriggered)};

                    std::vector<T> params;
                    e.params(params);

                    if (boost::math::tools::l2_norm(y) < eps
                        && (e.sympyStr() == "0.0" || e.sympyStr() == "0"))
                        cost = std::numeric_limits<T>::infinity();

                    results.emplace_back(Result{cost, e, std::chrono::milliseconds(0)});
                    expressions.emplace_back(e);

                    callback_(results.back());

                    if (cost < epsLoss)
                        break;
                }

                for (auto const& expression : extraExpressions_)
                {
                    Expression<T> e{expression};
                    auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_, timeoutTriggered)};

                    std::vector<T> params;
                    e.params(params);

                    if (boost::math::tools::l2_norm(y) < eps
                        && (e.sympyStr() == "0.0" || e.sympyStr() == "0"))
                        cost = std::numeric_limits<T>::infinity();

                    results.emplace_back(Result{cost, e, std::chrono::milliseconds(0)});
                    expressions.emplace_back(e);

                    callback_(results.back());

                    if (cost < epsLoss)
                        break;
                }

                {
                    std::sort(results.begin(), results.end(), [] (auto const& x, auto const& y) {return x.loss < y.loss;});

                    if (!results.empty() && results.front().loss < epsLoss)
                    {
                        io.stop();
                        io_thread.join();
                        
                        results_ = results;

                        return results.front();
                    }
                }

                std::map<size_t, std::vector<size_t> > unIndices;
                std::map<size_t, std::vector<std::pair<size_t, size_t> > > binIndices;

                for (size_t i{0}; i < un_ops_.size(); ++i)
                    unIndices[i] = std::vector<size_t>{};

                for (size_t i{0}; i < bin_ops_.size(); ++i)
                    binIndices[i] = std::vector<std::pair<size_t, size_t> >{};

                auto const yNull{boost::math::tools::l2_norm(y) < eps};
                T bestLoss{std::numeric_limits<T>::infinity()};

                for (size_t i{0}; i < niterations_; ++i)
                {
                    if (timeoutTriggered)
                        break;

                    auto function = [this, y, yNull, &timeoutTriggered, &keepBestCount, now, &bestLoss] (Expression<T> e) -> Result
                                    {
                                        auto cost{e.fit(y, paramValues_, epsLoss, verbose_, exhaustiveLimit, discreteParams_, timeoutTriggered)};

                                        if (yNull && cost < epsLoss)
                                        {
                                            if (e.symExpr().expand().simplify().isNull(eps))
                                                cost = std::numeric_limits<T>::infinity();
                                        }

                                        Result const r{cost, e, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - now)};

                                        callback_(r);

                                        if (cost < epsLoss)
                                            timeoutTriggered = true;

                                        if (++keepBestCount >= keepBestLimit)
                                            timeoutTriggered = true;

                                        if (cost < bestLoss)
                                        {
                                            bestLoss = cost;
                                            keepBestCount = 0;
                                        }

                                        return r;
                                    };

                    {
                        size_t const n{expressions.size()};
                        std::vector<std::future<Result> > futures;

                        for (size_t j = 0; j < n; ++j)
                        {
                            if (timeoutTriggered)
                                break;

                            for (size_t k = 0; k < un_ops_.size(); ++k)
                            {
                                if (timeoutTriggered)
                                    break;

                                auto const it{std::find(unIndices[k].begin(), unIndices[k].end(), j)};

                                if (it == unIndices[k].end())
                                {
                                    unIndices[k].emplace_back(j);

                                    auto const count{std::count(expressions[j].opTree().begin(), expressions[j].opTree().end(), un_ops_[k].name())};
                                    auto maxCount{std::numeric_limits<int>::max()};

                                    if (operatorDepth_.find(un_ops_[k].name()) != operatorDepth_.end())
                                        maxCount = operatorDepth_[un_ops_[k].name()];

                                    if (count < maxCount)
                                        futures.emplace_back(std::async(std::launch::async, function, Expression<T>{un_ops_[k], expressions[j]}));
                                }
                            }
                        }

                        for (auto& f : futures)
                        {
                            results.emplace_back(f.get());
                            expressions.emplace_back(results.back().expression);
                        }
                    }

                    {
                        std::sort(results.begin(), results.end(), [] (auto const& x, auto const& y) {return x.loss < y.loss;});

                        if (!results.empty() && results.front().loss < epsLoss)
                        {
                            io.stop();
                            io_thread.join();
                            
                            results_ = results;

                            return results.front();
                        }
                    }

                    {
                        size_t const n{expressions.size()};
                        std::vector<std::future<Result> > futures;

                        for (size_t j1 = 0; j1 < n; ++j1)
                        {
                            if (timeoutTriggered)
                                break;

                            for (size_t k = 0; k < bin_ops_.size(); ++k)
                            {
                                if (timeoutTriggered)
                                    break;

                                size_t j2{0};

                                if (bin_ops_[k].symmetry == BinaryOperator<T>::NonStrictSymmetry)
                                    j2 = j1;
                                else if (bin_ops_[k].symmetry == BinaryOperator<T>::StrictSymmetry)
                                    j2 = j1 + 1;

                                for (; j2 < n; ++j2)
                                {
                                    if (timeoutTriggered)
                                        break;

                                    auto const it{std::find(binIndices[k].begin(), binIndices[k].end(), std::make_pair(j1, j2))};

                                    if (it == binIndices[k].end())
                                    {
                                        binIndices[k].emplace_back(j1, j2);

                                        auto const count1{std::count(expressions[j1].opTree().begin(), expressions[j1].opTree().end(), bin_ops_[k].name())};
                                        auto const count2{std::count(expressions[j2].opTree().begin(), expressions[j2].opTree().end(), bin_ops_[k].name())};
                                        auto maxCount{std::numeric_limits<int>::max()};

                                        if (operatorDepth_.find(bin_ops_[k].name()) != operatorDepth_.end())
                                            maxCount = operatorDepth_[bin_ops_[k].name()];

                                        if (count1 < maxCount && count2 < maxCount)
                                            futures.emplace_back(std::async(std::launch::async, function, Expression<T>{bin_ops_[k], expressions[j1], expressions[j2]}));
                                    }
                                }
                            }
                        }

                        for (auto& f : futures)
                        {
                            results.emplace_back(f.get());
                            expressions.emplace_back(results.back().expression);
                        }
                    }

                    {
                        std::sort(results.begin(), results.end(), [] (auto const& x, auto const& y) {return x.loss < y.loss;});

                        if (!results.empty() && results.front().loss < epsLoss)
                        {
                            io.stop();
                            io_thread.join();

                            results_ = results;

                            return results.front();
                        }
                    }
                }

                io.stop();
                io_thread.join();

                results_ = results;

                if (results.empty())
                    throw std::runtime_error("No expression found!");

                return results_.front();
            }

            bool isTimeout() const
            {
                return isTimeout_;
            }
            
            std::vector<Result> const& results() const
            {
                return results_;
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
            std::function<void(Result const&)> callback_;
            bool discreteParams_;
            size_t timeout_;
            bool isTimeout_{false};
            std::vector<Result> results_;
    };
}

#endif // SYMREG_SYMBOLICREGRESSOR_H
