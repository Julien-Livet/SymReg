#ifndef BINARYOPERATOR_H
#define BINARYOPERATOR_H

#include <functional>
#include <limits>
#include <string>

#include <Eigen/Dense>

template <typename T>
class BinaryOperator
{
    public:
        enum Symmetry
        {
            NoSymmetry = 0,
            NonStrictSymmetry= 1,
            StrictSymmetry = 2
        };

        Symmetry symmetry;
        
        BinaryOperator(std::string const& name,
                       std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>,
                                                                        Eigen::Array<T, Eigen::Dynamic, 1>)> const& op,
                       Symmetry const& s = NoSymmetry) :
            symmetry{s}, name_{name}, op_{op}
        {
        }

        std::string const& name() const
        {
            return name_;
        }
        
        std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Array<T, Eigen::Dynamic, 1>)> op() const
        {
            return op_;
        }

        static BinaryOperator plus()
        {
            return BinaryOperator{"+", [] (auto const& x, auto const& y) {return x + y;}, StrictSymmetry};
        }

        static BinaryOperator minus()
        {
            return BinaryOperator{"-", [] (auto const& x, auto const& y) {return x - y;}, StrictSymmetry};
        }

        static BinaryOperator times()
        {
            return BinaryOperator{"*", [] (auto const& x, auto const& y) {return x *y;}, NonStrictSymmetry};
        }

        static BinaryOperator divide()
        {
            return BinaryOperator{"/", [] (auto const& x, auto const& y) {return x / y;}};
        }

        static BinaryOperator min()
        {
            return BinaryOperator{"min", [] (auto const& x, auto const& y) {return x.min(y);}, StrictSymmetry};
        }

        static BinaryOperator max()
        {
            return BinaryOperator{"max", [] (auto const& x, auto const& y) {return x.max(y);}, StrictSymmetry};
        }

    private:
        std::string name_;
        std::function<Eigen::Array<T, Eigen::Dynamic, 1>(Eigen::Array<T, Eigen::Dynamic, 1>, Eigen::Array<T, Eigen::Dynamic, 1>)> op_;
};

#endif // BINARYOPERATOR_H
