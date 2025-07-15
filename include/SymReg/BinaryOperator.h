#ifndef SYMREG_BINARYOPERATOR_H
#define SYMREG_BINARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include <ceres/ceres.h>

#include <Sym/Expression.h>

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
                           std::function<sym::Expression<T>(sym::Expression<T> const&, sym::Expression<T> const&)> const& symOp,
                           Symmetry const& s = NoSymmetry) :
                symmetry{s}, name_{name}, op_{op}, jetOp_{jetOp}, symOp_{symOp}
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
                                      [] (auto const& x, auto const& y) {return sym::min(x, y);},
                                      StrictSymmetry};
            }

            static BinaryOperator max()
            {
                return BinaryOperator{"max",
                                      [] (auto const& x, auto const& y) {return x.max(y);},
                                      [] (auto const& x, auto const& y) {return x.max(y);},
                                      [] (auto const& x, auto const& y) {return sym::max(x, y);},
                                      StrictSymmetry};
            }

            static BinaryOperator pow()
            {
                return BinaryOperator{"pow",
                                      [] (auto const& x, auto const& y) {return x.pow(y);},
                                      [] (auto const& x, auto const& y) {return x.pow(y);},
                                      [] (auto const& x, auto const& y) {return sym::Mul(x, y);}};
            }

            bool operator==(BinaryOperator<T> const& other) const
            {
                return name_ == other.name_;
            }

        private:
            std::string name_;
            std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
            std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>, Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> jetOp_;
            std::function<sym::Expression<T>(sym::Expression<T> const&, sym::Expression<T> const&)> symOp_;
    };
}

#endif // SYMREG_BINARYOPERATOR_H
