#ifndef UNARYOPERATOR_H
#define UNARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

template <typename T>
class UnaryOperator
{
    public:
        UnaryOperator(std::string const& name,
                      std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> const& op) :
            name_{name}, op_{op}
        {
        }

        std::string const& name() const
        {
            return name_;
        }

        std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> op() const
        {
            return op_;
        }

        static UnaryOperator log()
        {
            return UnaryOperator{"log", [] (auto const& x) {return Eigen::log(x);}};
        }

        static UnaryOperator exp()
        {
            return UnaryOperator{"exp", [] (auto const& x) {return Eigen::exp(x);}};
        }

        static UnaryOperator cos()
        {
            return UnaryOperator{"cos", [] (auto const& x) {return Eigen::cos(x);}};
        }

        static UnaryOperator sin()
        {
            return UnaryOperator{"sin", [] (auto const& x) {return Eigen::sin(x);}};
        }

        static UnaryOperator tan()
        {
            return UnaryOperator{"tan", [] (auto const& x) {return Eigen::tan(x);}};
        }

        static UnaryOperator acos()
        {
            return UnaryOperator{"acos", [] (auto const& x) {return Eigen::acos(x);}};
        }

        static UnaryOperator asin()
        {
            return UnaryOperator{"asin", [] (auto const& x) {return Eigen::asin(x);}};
        }

        static UnaryOperator atan()
        {
            return UnaryOperator{"tan", [] (auto const& x) {return Eigen::atan(x);}};
        }

        static UnaryOperator cosh()
        {
            return UnaryOperator{"cosh", [] (auto const& x) {return Eigen::cosh(x);}};
        }

        static UnaryOperator sinh()
        {
            return UnaryOperator{"sinh", [] (auto const& x) {return Eigen::sinh(x);}};
        }

        static UnaryOperator tanh()
        {
            return UnaryOperator{"tanh", [] (auto const& x) {return Eigen::tanh(x);}};
        }

        static UnaryOperator acosh()
        {
            return UnaryOperator{"acosh", [] (auto const& x) {return Eigen::acosh(x);}};
        }

        static UnaryOperator asinh()
        {
            return UnaryOperator{"asinh", [] (auto const& x) {return Eigen::asinh(x);}};
        }

        static UnaryOperator atanh()
        {
            return UnaryOperator{"atanh", [] (auto const& x) {return Eigen::atanh(x);}};
        }

        static UnaryOperator sqrt()
        {
            return UnaryOperator{"sqrt", [] (auto const& x) {return Eigen::sqrt(x);}};
        }

        static UnaryOperator floor()
        {
            return UnaryOperator{"floor", [] (auto const& x) {return Eigen::floor(x);}};
        }

        static UnaryOperator ceil()
        {
            return UnaryOperator{"ceil", [] (auto const& x) {return Eigen::ceil(x);}};
        }

        static UnaryOperator abs()
        {
            return UnaryOperator{"abs", [] (auto const& x) {return Eigen::abs(x);}};
        }

        static UnaryOperator inverse()
        {
            return UnaryOperator{"inverse", [] (auto const& x) {return Eigen::inverse(x);}};
        }

    private:
        std::string name_;
        std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
};

#endif // UNARYOPERATOR_H
