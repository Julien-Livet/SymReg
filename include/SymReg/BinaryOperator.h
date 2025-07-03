#ifndef BINARYOPERATOR_H
#define BINARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

#include <ceres/ceres.h>

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
                           Symmetry const& s = NoSymmetry) :
                symmetry{s}, name_{name}, op_{op}, jetOp_{jetOp}
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

            auto const jetOp() const
            {
                return jetOp_;
            }

            static BinaryOperator plus()
            {
                return BinaryOperator{"+",
                                      [] (auto const& x, auto const& y) {return x + y;},
                                      [] (auto const& x, auto const& y) {return x + y;},
                                      StrictSymmetry};
            }

            static BinaryOperator minus()
            {
                return BinaryOperator{"-",
                                      [] (auto const& x, auto const& y) {return x - y;},
                                      [] (auto const& x, auto const& y) {return x - y;},
                                      StrictSymmetry};
            }

            static BinaryOperator times()
            {
                return BinaryOperator{"*",
                                      [] (auto const& x, auto const& y) {return x * y;},
                                      [] (auto const& x, auto const& y) {return x * y;},
                                      NonStrictSymmetry};
            }

            static BinaryOperator divide()
            {
                return BinaryOperator{"/",
                                      [] (auto const& x, auto const& y) {return x / y;},
                                      [] (auto const& x, auto const& y) {return x / y;}};
            }

            static BinaryOperator min()
            {
                return BinaryOperator{"min",
                                      [] (auto const& x, auto const& y) {return x.min(y);},
                                      [] (auto const& x, auto const& y) {return x.min(y);},
                                      StrictSymmetry};
            }

            static BinaryOperator max()
            {
                return BinaryOperator{"max",
                                      [] (auto const& x, auto const& y) {return x.max(y);},
                                      [] (auto const& x, auto const& y) {return x.max(y);},
                                      StrictSymmetry};
            }

            static BinaryOperator pow()
            {
                return BinaryOperator{"pow",
                                      [] (auto const& x, auto const& y) {return x.pow(y);},
                                      [] (auto const& x, auto const& y) {return x.pow(y);}};
            }
            
            bool operator==(BinaryOperator<T> const& other) const
            {
                return name_ == other.name_;
            }

        private:
            std::string name_;
            std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
            std::function<Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>(Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>, Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1>)> jetOp_;
    };
}

#endif // BINARYOPERATOR_H
