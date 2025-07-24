#ifndef SYMREG_UNARYOPERATOR_H
#define SYMREG_UNARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <Sym/Expression.h>

namespace sr
{
    template <typename T>
    class UnaryOperator
    {
        public:
            UnaryOperator(std::string const& name,
                          std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> const& op,
                          std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> const& jetOp,
                          std::function<sym::Expression<T>(sym::Expression<T> const&)> const& symOp) :
                name_{name}, op_{op}, jetOp_{jetOp}, symOp_{symOp}
            {
            }

            std::string const& name() const
            {
                return name_;
            }

            auto const& op() const
            {
                return op_;
            }

            auto const& jetOp() const
            {
                return jetOp_;
            }

            auto const& symOp() const
            {
                return symOp_;
            }

            static UnaryOperator log()
            {
                return UnaryOperator{"log",
                                     [] (auto const& x) {return x.log();},
                                     [] (auto const& x) {return x.log();},
                                     [] (auto const& x) {return sym::log(x);}};
            }

            static UnaryOperator exp()
            {
                return UnaryOperator{"exp",
                                     [] (auto const& x) {return x.exp();},
                                     [] (auto const& x) {return x.exp();},
                                     [] (auto const& x) {return sym::exp(x);}};
            }

            static UnaryOperator cot()
            {
                return UnaryOperator{"cot",
                                     [] (auto const& x) {return x.tan().inverse();},
                                     [] (auto const& x) {return x.tan().inverse();},
                                     [] (auto const& x) {return sym::cot(x);}};
            }

            static UnaryOperator cos()
            {
                return UnaryOperator{"cos",
                                     [] (auto const& x) {return x.cos();},
                                     [] (auto const& x) {return x.cos();},
                                     [] (auto const& x) {return sym::cos(x);}};
            }

            static UnaryOperator sin()
            {
                return UnaryOperator{"sin",
                                     [] (auto const& x) {return x.sin();},
                                     [] (auto const& x) {return x.sin();},
                                     [] (auto const& x) {return sym::sin(x);}};
            }

            static UnaryOperator tan()
            {
                return UnaryOperator{"tan",
                                     [] (auto const& x) {return x.tan();},
                                     [] (auto const& x) {return x.tan();},
                                     [] (auto const& x) {return sym::tan(x);}};
            }

            static UnaryOperator acos()
            {
                return UnaryOperator{"acos",
                                     [] (auto const& x) {return x.acos();},
                                     [] (auto const& x) {return x.acos();},
                                     [] (auto const& x) {return sym::acos(x);}};
            }

            static UnaryOperator asin()
            {
                return UnaryOperator{"asin",
                                     [] (auto const& x) {return x.asin();},
                                     [] (auto const& x) {return x.asin();},
                                     [] (auto const& x) {return sym::asin(x);}};
            }

            static UnaryOperator atan()
            {
                return UnaryOperator{"atan",
                                     [] (auto const& x) {return x.atan();},
                                     [] (auto const& x) {return x.atan();},
                                     [] (auto const& x) {return sym::atan(x);}};
            }

            static UnaryOperator cosh()
            {
                return UnaryOperator{"cosh",
                                     [] (auto const& x) {return x.cosh();},
                                     [] (auto const& x) {return x.cosh();},
                                     [] (auto const& x) {return sym::cosh(x);}};
            }

            static UnaryOperator sinh()
            {
                return UnaryOperator{"sinh",
                                     [] (auto const& x) {return x.sinh();},
                                     [] (auto const& x) {return x.sinh();},
                                     [] (auto const& x) {return sym::sinh(x);}};
            }

            static UnaryOperator tanh()
            {
                return UnaryOperator{"tanh",
                                     [] (auto const& x) {return x.tanh();},
                                     [] (auto const& x) {return x.tanh();},
                                     [] (auto const& x) {return sym::tanh(x);}};
            }
/*
            static UnaryOperator acosh()
            {
                return UnaryOperator{"acosh",
                                     [] (auto const& x) {return x.acosh();},
                                     [] (auto const& x) {return x.acosh();},
                                     [] (auto const& x) {return sym::acosh(x);}};
            }

            static UnaryOperator asinh()
            {
                return UnaryOperator{"asinh",
                                     [] (auto const& x) {return x.asinh();},
                                     [] (auto const& x) {return x.asinh();},
                                     [] (auto const& x) {return sym::asinh(x);}};
            }

            static UnaryOperator atanh()
            {
                return UnaryOperator{"atanh",
                                     [] (auto const& x) {return x.atanh();},
                                     [] (auto const& x) {return x.atanh();},
                                     [] (auto const& x) {return sym::atanh(x);}};
            }
*/
            static UnaryOperator sqrt()
            {
                return UnaryOperator{"sqrt",
                                     [] (auto const& x) {return x.sqrt();},
                                     [] (auto const& x) {return x.sqrt();},
                                     [] (auto const& x) {return sym::sqrt(x);}};
            }

            static UnaryOperator floor()
            {
                return UnaryOperator{"floor",
                                     [] (auto const& x) {return x.floor();},
                                     [] (auto const& x) {return x.floor();},
                                     [] (auto const& x) {return sym::floor(x);}};
            }

            static UnaryOperator ceil()
            {
                return UnaryOperator{"ceil",
                                     [] (auto const& x) {return x.ceil();},
                                     [] (auto const& x) {return x.ceil();},
                                     [] (auto const& x) {return sym::ceil(x);}};
            }

            static UnaryOperator abs()
            {
                return UnaryOperator{"abs",
                                     [] (auto const& x) {return x.abs();},
                                     [] (auto const& x) {return x.abs();},
                                     [] (auto const& x) {return sym::abs(x);}};
            }

            static UnaryOperator inverse()
            {
                return UnaryOperator{"inverse",
                                     [] (auto const& x) {return x.inverse();},
                                     [] (auto const& x) {return x.inverse();},
                                     [] (auto const& x) {return sym::inverse(x);}};
            }

            bool operator==(UnaryOperator<T> const& other) const
            {
                return name_ == other.name_;
            }

        private:
            std::string name_;
            std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
            std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> jetOp_;
            std::function<sym::Expression<T>(sym::Expression<T> const&)> symOp_;
    };
}

#endif // SYMREG_UNARYOPERATOR_H
