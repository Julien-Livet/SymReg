#ifndef SYMBOLICREGRESSOR_H
#define SYMBOLICREGRESSOR_H

#include <limits>
#include <vector>

#include "BinaryOperator.h"
#include "Expression.h"
#include "UnaryOperator.h"
#include "Variable.h"

template <typename T>
class SymbolicRegressor
{
    public:
        T eps = std::numeric_limits<T>::epsilon();
        T epsLoss = std::numeric_limits<T>::epsilon();

        SymbolicRegressor(std::vector<Variable<T> > const& variables,
                          std::vector<UnaryOperator<T> > const& un_ops,
                          std::vector<BinaryOperator<T> > const& bin_ops)
            : variables_{variables}, un_ops_{un_ops}, bin_ops_{bin_ops}
        {
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
        
        void fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y)
        {
            std::vector<Expression<T> > expressions;
            std::vector<T> costs;

            for (auto const& v : variables_)
            {
                expressions.emplace_back(v);
                costs.emplace_back(expressions.back().fit(y));
/*
                std::vector<T> params;
                expressions.back().params(params);
                for (auto const& v: params)
                    std::cout << v << " ";
                std::cout << std::endl;
                std::cout << costs.back() << std::endl;*/
            }
        }

    private:
        std::vector<Variable<T> > variables_;
        std::vector<UnaryOperator<T> > un_ops_;
        std::vector<BinaryOperator<T> > bin_ops_;
};

#endif // SYMBOLICREGRESSOR_H
