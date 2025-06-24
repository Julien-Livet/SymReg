#ifndef EXPRESSION_H
#define EXPRESSION_H

#include <any>

#include <ceres/ceres.h>

#include "BinaryOperator.h"
#include "UnaryOperator.h"
#include "Variable.h"

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
    
        Expression(Variable<T> const& variable) : operand1_{variable}, operand2_{0}, op_{0}
        {
        }
        
        Expression(UnaryOperator<T> const& op, Expression const& operand) : operand1_{operand}, operand2_{0}, op_{op}
        {
        }
        
        Expression(BinaryOperator<T> const& op, Expression const& operand1, Expression const& operand2)
            : operand1_{operand1}, operand2_{operand2}, op_{op}
        {
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

        T fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y)
        {
            std::vector<double> params;
            this->params(params);
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
            options.minimizer_progress_to_stdout = true;

            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            applyParams(params);

            std::cout << summary.BriefReport() << "\n";
            for (auto const& v: params)
                std::cout << v << " ";
            std::cout << std::endl;
            std::cout << summary.final_cost << std::endl;

            return summary.final_cost;
        }
        
    private:
        std::any operand1_; //Variable<T> or Expression
        std::any operand2_; //0 or valid operand
        std::any op_; //0 or valid operator
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
    Residual(Expression<T>& expression,
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

        for (size_t i{0}; i < params.size(); ++i)
        {
            if constexpr (typeid(S) == typeid(T))
                params[i] = *p[i];
            else
                params[i] = p[i]->a;
        }

        expression_.applyParams(params);

        auto const x{expression_.eval()};

        for (int i{0}; i < x.size(); ++i)
            residual[i] = S(y_[i]) - S(x[i]);

        return true;
    }

    private:
        Expression<T>& expression_;
        const Eigen::Array<double, Eigen::Dynamic, 1> y_;
};

#endif // EXPRESSION_H
