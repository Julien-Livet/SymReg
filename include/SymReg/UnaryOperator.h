#ifndef UNARYOPERATOR_H
#define UNARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <ginac/ginac.h>

namespace GiNaC
{
    DECLARE_FUNCTION_1P(inverse)

    GiNaC::ex inverse_eval(GiNaC::ex const& a)
    {
        if (is_a<GiNaC::numeric>(a))
            return 1 / a;

        return inverse(a);
    }

    REGISTER_FUNCTION(inverse, eval_func(inverse_eval))

    DECLARE_FUNCTION_1P(floor)

    GiNaC::ex floor_eval(GiNaC::ex const& a)
    {
        if (is_a<GiNaC::numeric>(a))
            return std::floor(GiNaC::ex_to<GiNaC::numeric>(a).to_double());

        return floor(a);
    }

    REGISTER_FUNCTION(floor, eval_func(floor_eval))

    DECLARE_FUNCTION_1P(ceil)

    GiNaC::ex ceil_eval(GiNaC::ex const& a)
    {
        if (is_a<GiNaC::numeric>(a))
            return std::ceil(GiNaC::ex_to<GiNaC::numeric>(a).to_double());

        return ceil(a);
    }

    REGISTER_FUNCTION(ceil, eval_func(ceil_eval))
}

namespace sr
{
    template <typename T>
    class UnaryOperator
    {
        public:
            UnaryOperator(std::string const& name,
                          std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> const& op,
                          std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> const& jetOp,
                          std::function<GiNaC::ex(GiNaC::ex const&)> const& ginacOp) :
                name_{name}, op_{op}, jetOp_{jetOp}, ginacOp_{ginacOp}
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

            auto const& ginacOp() const
            {
                return ginacOp_;
            }

            static UnaryOperator log()
            {
                return UnaryOperator{"log",
                                     [] (auto const& x) {return x.log();},
                                     [] (auto const& x) {return x.log();},
                                     [] (auto const& x) {return GiNaC::log(x);}};
            }

            static UnaryOperator exp()
            {
                return UnaryOperator{"exp",
                                     [] (auto const& x) {return x.exp();},
                                     [] (auto const& x) {return x.exp();},
                                     [] (auto const& x) {return GiNaC::exp(x);}};
            }

            static UnaryOperator cot()
            {
                return UnaryOperator{"cot",
                                     [] (auto const& x) {return x.tan().inverse();},
                                     [] (auto const& x) {return x.tan().inverse();},
                                     [] (auto const& x) {return GiNaC::inverse(GiNaC::tan(x));}};
            }

            static UnaryOperator cos()
            {
                return UnaryOperator{"cos",
                                     [] (auto const& x) {return x.cos();},
                                     [] (auto const& x) {return x.cos();},
                                     [] (auto const& x) {return GiNaC::cos(x);}};
            }

            static UnaryOperator sin()
            {
                return UnaryOperator{"sin",
                                     [] (auto const& x) {return x.sin();},
                                     [] (auto const& x) {return x.sin();},
                                     [] (auto const& x) {return GiNaC::sin(x);}};
            }

            static UnaryOperator tan()
            {
                return UnaryOperator{"tan",
                                     [] (auto const& x) {return x.tan();},
                                     [] (auto const& x) {return x.tan();},
                                     [] (auto const& x) {return GiNaC::tan(x);}};
            }

            static UnaryOperator acos()
            {
                return UnaryOperator{"acos",
                                     [] (auto const& x) {return x.acos();},
                                     [] (auto const& x) {return x.acos();},
                                     [] (auto const& x) {return GiNaC::acos(x);}};
            }

            static UnaryOperator asin()
            {
                return UnaryOperator{"asin",
                                     [] (auto const& x) {return x.asin();},
                                     [] (auto const& x) {return x.asin();},
                                     [] (auto const& x) {return GiNaC::asin(x);}};
            }

            static UnaryOperator atan()
            {
                return UnaryOperator{"atan",
                                     [] (auto const& x) {return x.atan();},
                                     [] (auto const& x) {return x.atan();},
                                     [] (auto const& x) {return GiNaC::atan(x);}};
            }

            static UnaryOperator cosh()
            {
                return UnaryOperator{"cosh",
                                     [] (auto const& x) {return x.cosh();},
                                     [] (auto const& x) {return x.cosh();},
                                     [] (auto const& x) {return GiNaC::cosh(x);}};
            }

            static UnaryOperator sinh()
            {
                return UnaryOperator{"sinh",
                                     [] (auto const& x) {return x.sinh();},
                                     [] (auto const& x) {return x.sinh();},
                                     [] (auto const& x) {return GiNaC::sinh(x);}};
            }

            static UnaryOperator tanh()
            {
                return UnaryOperator{"tanh",
                                     [] (auto const& x) {return x.tanh();},
                                     [] (auto const& x) {return x.tanh();},
                                     [] (auto const& x) {return GiNaC::tanh(x);}};
            }

            static UnaryOperator acosh()
            {
                return UnaryOperator{"acosh",
                                     [] (auto const& x) {return x.acosh();},
                                     [] (auto const& x) {return x.acosh();},
                                     [] (auto const& x) {return GiNaC::acosh(x);}};
            }

            static UnaryOperator asinh()
            {
                return UnaryOperator{"asinh",
                                     [] (auto const& x) {return x.asinh();},
                                     [] (auto const& x) {return x.asinh();},
                                     [] (auto const& x) {return GiNaC::asinh(x);}};
            }

            static UnaryOperator atanh()
            {
                return UnaryOperator{"atanh",
                                     [] (auto const& x) {return x.atanh();},
                                     [] (auto const& x) {return x.atanh();},
                                     [] (auto const& x) {return GiNaC::atanh(x);}};
            }

            static UnaryOperator sqrt()
            {
                return UnaryOperator{"sqrt",
                                     [] (auto const& x) {return x.sqrt();},
                                     [] (auto const& x) {return x.sqrt();},
                                     [] (auto const& x) {return GiNaC::sqrt(x);}};
            }

            static UnaryOperator floor()
            {
                return UnaryOperator{"floor",
                                     [] (auto const& x) {return x.floor();},
                                     [] (auto const& x) {return x.floor();},
                                     [] (auto const& x) {return GiNaC::floor(x);}};
            }

            static UnaryOperator ceil()
            {
                return UnaryOperator{"ceil",
                                     [] (auto const& x) {return x.ceil();},
                                     [] (auto const& x) {return x.ceil();},
                                     [] (auto const& x) {return GiNaC::ceil(x);}};
            }

            static UnaryOperator abs()
            {
                return UnaryOperator{"abs",
                                     [] (auto const& x) {return x.abs();},
                                     [] (auto const& x) {return x.abs();},
                                     [] (auto const& x) {return GiNaC::abs(x);}};
            }

            static UnaryOperator inverse()
            {
                return UnaryOperator{"inverse",
                                     [] (auto const& x) {return x.inverse();},
                                     [] (auto const& x) {return x.inverse();},
                                     [] (auto const& x) {return GiNaC::inverse(x);}};
            }
            
            bool operator==(UnaryOperator<T> const& other) const
            {
                return name_ == other.name_;
            }

        private:
            std::string name_;
            std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
            std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> jetOp_;
            std::function<GiNaC::ex(GiNaC::ex const&)> ginacOp_;
    };
}

#endif // UNARYOPERATOR_H
