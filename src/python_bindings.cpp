#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "SymReg/BinaryOperator.h"
#include "SymReg/Expression.h"
#include "SymReg/SymbolicRegressor.h"
#include "SymReg/UnaryOperator.h"
#include "SymReg/Variable.h"

namespace py = pybind11;

using namespace sr;

PYBIND11_MODULE(symreg, m)
{
    py::class_<Expression<double> >(m, "Expression")
        .def("str", &Expression<double>::str)
        .def("optStr", &Expression<double>::optStr)
        .def("dot", &Expression<double>::dot)
        .def_static("var", [] (Variable<double> const& v)
                           {
                               return Expression<double>{v};
                           })
        .def_static("un", [] (UnaryOperator<double> const& op, Expression<double> const& operand)
                          {
                              return Expression<double>{op, operand};
                          })
        .def_static("bin", [] (BinaryOperator<double> const& op, Expression<double> const& operand1, Expression<double> const& operand2)
                           {
                               return Expression<double>{op, operand1, operand2};
                           });

    py::class_<BinaryOperator<double> > binop(m, "BinaryOperator");

    py::enum_<BinaryOperator<double>::Symmetry>(binop, "Symmetry")
        .value("NoSymmetry", BinaryOperator<double>::Symmetry::NoSymmetry)
        .value("NonStrictSymmetry", BinaryOperator<double>::Symmetry::NonStrictSymmetry)
        .value("StrictSymmetry", BinaryOperator<double>::Symmetry::StrictSymmetry)
        .export_values();

    binop.def_static("plus", &BinaryOperator<double>::plus,
                    py::arg("symmetry") = BinaryOperator<double>::Symmetry::StrictSymmetry)
        .def_static("minus", &BinaryOperator<double>::minus,
                    py::arg("symmetry") = BinaryOperator<double>::Symmetry::StrictSymmetry)
        .def_static("times", &BinaryOperator<double>::times)
        .def_static("divide", &BinaryOperator<double>::divide)
        .def_static("min", &BinaryOperator<double>::min)
        .def_static("max", &BinaryOperator<double>::max)
        .def_static("pow", &BinaryOperator<double>::pow);

    py::class_<UnaryOperator<double> >(m, "UnaryOperator")
        .def_static("log", &UnaryOperator<double>::log)
        .def_static("exp", &UnaryOperator<double>::exp)
        .def_static("cot", &UnaryOperator<double>::cot)
        .def_static("cos", &UnaryOperator<double>::cos)
        .def_static("sin", &UnaryOperator<double>::sin)
        .def_static("tan", &UnaryOperator<double>::tan)
        .def_static("acos", &UnaryOperator<double>::acos)
        .def_static("asin", &UnaryOperator<double>::asin)
        .def_static("atan", &UnaryOperator<double>::atan)
        .def_static("cosh", &UnaryOperator<double>::cosh)
        .def_static("sinh", &UnaryOperator<double>::sinh)
        .def_static("tanh", &UnaryOperator<double>::tanh)/*
        .def_static("acosh", &UnaryOperator<double>::acosh)
        .def_static("asinh", &UnaryOperator<double>::asinh)
        .def_static("atanh", &UnaryOperator<double>::atanh)*/
        .def_static("sqrt", &UnaryOperator<double>::sqrt)
        .def_static("floor", &UnaryOperator<double>::floor)
        .def_static("ceil", &UnaryOperator<double>::ceil)
        .def_static("abs", &UnaryOperator<double>::abs)
        .def_static("inverse", &UnaryOperator<double>::inverse);

    py::class_<Variable<double> >(m, "Variable")
        .def(py::init<std::string const&,
                      Eigen::Array<double, Eigen::Dynamic, 1> const&>());

    py::class_<SymbolicRegressor<double> > sr(m, "SymbolicRegressor");

    py::class_<SymbolicRegressor<double>::Result>(sr, "Result")
        .def_readwrite("loss", &SymbolicRegressor<double>::Result::loss)
        .def_readwrite("expression", &SymbolicRegressor<double>::Result::expression)
        .def_readwrite("time", &SymbolicRegressor<double>::Result::time);

    sr.def(py::init<std::vector<Variable<double> > const& ,
                    std::vector<UnaryOperator<double> > const&,
                    std::vector<BinaryOperator<double> > const&,
                    size_t,
                    std::vector<double> const&,
                    std::map<std::string, size_t> const&,
                    std::vector<Expression<double> > const&,
                    bool,
                    std::function<void(SymbolicRegressor<double>::Result const&)> const&,
                    bool,
                    size_t,
                    size_t>(),
                    py::arg("variables"),
                    py::arg("un_ops") = std::vector<UnaryOperator<double> >{},
                    py::arg("bin_ops") = std::vector<BinaryOperator<double> >{},
                    py::arg("niterations") = 5,
                    py::arg("operatorDepth") = std::map<std::string, size_t>{},
                    py::arg("paramValues") = std::vector<double>{},
                    py::arg("extraExpressions") = std::vector<Expression<double> >{},
                    py::arg("verbose") = false,
                    py::arg("callback") = nullptr,
                    py::arg("discreteParams") = true,
                    py::arg("timeout") = 30 * 60,
                    py::arg("keepBestLimit") = 10000)
        .def_readwrite("eps", &SymbolicRegressor<double>::eps)
        .def_readwrite("epsLoss", &SymbolicRegressor<double>::epsLoss)
        .def_readwrite("exhaustiveLimit", &SymbolicRegressor<double>::exhaustiveLimit)
        .def_readwrite("keepBestLimit", &SymbolicRegressor<double>::keepBestLimit)
        .def("fit", [] (SymbolicRegressor<double>& self,
                        Eigen::ArrayXd const& y)
                    {
                        py::gil_scoped_release release;
                        
                        return self.fit(y);
                    })
        .def("results", &SymbolicRegressor<double>::results)
        .def("isTimeout", &SymbolicRegressor<double>::isTimeout);
}
