#ifndef BINARYOPERATOR_H
#define BINARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <ginac/ginac.h>

namespace GiNaC
{
    DECLARE_FUNCTION_2P(min)

    GiNaC::ex min_eval(GiNaC::ex const& a, GiNaC::ex const& b)
    {
        if (GiNaC::is_a<GiNaC::numeric>(a) && GiNaC::is_a<GiNaC::numeric>(b))
            return GiNaC::ex_to<GiNaC::numeric>(a) < GiNaC::ex_to<GiNaC::numeric>(b) ? a : b;

        return min(a, b);
    }

    REGISTER_FUNCTION(min, eval_func(min_eval))

    DECLARE_FUNCTION_2P(max)

    GiNaC::ex max_eval(GiNaC::ex const& a, GiNaC::ex const& b)
    {
        if (GiNaC::is_a<GiNaC::numeric>(a) && GiNaC::is_a<GiNaC::numeric>(b))
            return GiNaC::ex_to<GiNaC::numeric>(a) < GiNaC::ex_to<GiNaC::numeric>(b) ? b : a;

        return max(a, b);
    }

    REGISTER_FUNCTION(max, eval_func(max_eval))
}

namespace sr
{
    template <typename T>
    class BinaryOperator
    {
        public:
            enum Symmetry
            {
                NoSymmetry = 0,
                NonStrictSymmetry = 1,
                StrictSymmetry = 2
            };

            Symmetry symmetry;

            BinaryOperator(std::string const& name,
                           std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>,
                                                                            Eigen::Array<T, Eigen::Dynamic, 1>)> const& op,
                           std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>,
                                                                                           Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> const& jetOp,
                           std::function<GiNaC::ex(GiNaC::ex const&, GiNaC::ex const&)> const& ginacOp,
                           Symmetry const& s = NoSymmetry) :
                symmetry{s}, name_{name}, op_{op}, jetOp_{jetOp}, ginacOp_{ginacOp}
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

            static BinaryOperator plus(Symmetry symmetry = StrictSymmetry)
            {
                return BinaryOperator{"+",
                                      [] (auto const& x, auto const& y) {return x + y;},
                                      [] (auto const& x, auto const& y) {return x + y;},
                                      [] (auto const& x, auto const& y) {return x + y;},
                                      symmetry};
            }

            static BinaryOperator minus(Symmetry symmetry = StrictSymmetry)
            {
                return BinaryOperator{"-",
                                      [] (auto const& x, auto const& y) {return x - y;},
                                      [] (auto const& x, auto const& y) {return x - y;},
                                      [] (auto const& x, auto const& y) {return x - y;},
                                      symmetry};
            }

            static BinaryOperator times()
            {
                return BinaryOperator{"*",
                                      [] (auto const& x, auto const& y) {return x * y;},
                                      [] (auto const& x, auto const& y) {return x * y;},
                                      [] (auto const& x, auto const& y) {return x * y;},
                                      NonStrictSymmetry};
            }

            static BinaryOperator divide()
            {
                return BinaryOperator{"/",
                                      [] (auto const& x, auto const& y) {return x / y;},
                                      [] (auto const& x, auto const& y) {return x / y;},
                                      [] (auto const& x, auto const& y) {return x / y;}};
            }

            static BinaryOperator min()
            {
                return BinaryOperator{"min",
                                      [] (auto const& x, auto const& y) {return x.min(y);},
                                      [] (auto const& x, auto const& y) {return x.min(y);},
                                      [] (auto const& x, auto const& y) {return GiNaC::min(x, y);},
                                      StrictSymmetry};
            }

            static BinaryOperator max()
            {
                return BinaryOperator{"max",
                                      [] (auto const& x, auto const& y) {return x.max(y);},
                                      [] (auto const& x, auto const& y) {return x.max(y);},
                                      [] (auto const& x, auto const& y) {return GiNaC::max(x, y);},
                                      StrictSymmetry};
            }

            static BinaryOperator pow()
            {
                return BinaryOperator{"pow",
                                      [] (auto const& x, auto const& y) {return x.pow(y);},
                                      [] (auto const& x, auto const& y) {return x.pow(y);},
                                      [] (auto const& x, auto const& y) {return GiNaC::pow(x, y);}};
            }

            bool operator==(BinaryOperator<T> const& other) const
            {
                return name_ == other.name_;
            }

        private:
            std::string name_;
            std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
            std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>, Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> jetOp_;
            std::function<GiNaC::ex(GiNaC::ex const&, GiNaC::ex const&)> const& ginacOp_;
    };
}

#endif // BINARYOPERATOR_H
