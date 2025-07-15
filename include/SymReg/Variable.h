#ifndef SYMREG_VARIABLE_H
#define SYMREG_VARIABLE_H

#include <limits>
#include <string>

#include <Eigen/Dense>

namespace sr
{
    template <typename T>
    class Variable
    {
        public:
            using Type = T;

            Variable(std::string const& name, Eigen::Array<T, Eigen::Dynamic, 1> const& value)
                : name_{name}, value_{value}
            {
            }

            std::string const& name() const
            {
                return name_;
            }

            Eigen::Array<T, Eigen::Dynamic, 1> const& value() const
            {
                return value_;
            }
            
            bool operator==(Variable<T> const& other) const
            {
                return name_ == other.name_;
            }

        private:
            std::string name_;
            Eigen::Array<T, Eigen::Dynamic, 1> value_;
    };
}

#endif // SYMREG_VARIABLE_H
