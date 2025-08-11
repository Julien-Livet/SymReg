#ifndef SYMREG_EXPRESSION_H
#define SYMREG_EXPRESSION_H

#include <algorithm>
#include <any>
#include <deque>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <ceres/ceres.h>

#include <pybind11/embed.h>

#include "SymReg/BinaryOperator.h"
#include "SymReg/UnaryOperator.h"
#include "SymReg/Variable.h"

namespace sr
{
    struct Node
    {
        std::string label;
        std::string color;
        std::string tooltip;
        std::vector<Node*> children;

        Node(std::string const& label = std::string{},
             std::string const& color = std::string{},
             std::string const& tooltip = std::string{},
             std::vector<Node*> const& children = std::vector<Node*>{}) : label{label}, color{color}, tooltip{tooltip}, children{children}
        {
        }
    };

    void freeDot(Node* node)
    {
        for (auto const& c: node->children)
        {
            freeDot(c);
            delete c;
        }

        node->children.clear();
    }

    void writeDot(Node* node, std::ostream& out, size_t& counter, std::string const& parent = "")
    {
        std::string current = "n" + std::to_string(counter++);

        out << "    " << current << " [label=\"" << node->label << "\", shape=circle, style=filled, fillcolor=" << node->color << ", tooltip=\"" << node->tooltip << "\"];\n";

        if (!parent.empty())
            out << "    " << parent << " -> " << current << ";\n";

        for (auto* child: node->children)
            writeDot(child, out, counter, current);
    }

    bool isSymbol(std::string const& s)
    {
        return s == "+" || s == "-" || s == "*" || s == "/";
    }

    template<typename T>
    struct EigenArrayLess
    {
        bool operator()(Eigen::Array<T, Eigen::Dynamic, 1> const& a,
                        Eigen::Array<T, Eigen::Dynamic, 1> const& b) const
        {
            const auto size_a = a.size();
            const auto size_b = b.size();

            if (size_a != size_b)
                return size_a < size_b;

            for (int i = 0; i < size_a; ++i)
            {
                if (a[i] < b[i])
                    return true;

                if (a[i] > b[i])
                    return false;
            }

            return false;
        }
    };

    template <typename T>
    std::string to_string_c_locale(T value)
    {
        std::ostringstream oss;

        oss.imbue(std::locale::classic());
        oss << value;

        return oss.str();
    }

    template <typename T>
    std::vector<std::vector<T > > discreteValues(size_t n, size_t m, std::vector<T> const& paramValues)
    {
        std::vector<std::vector<T> > values;
        values.reserve(n);

        std::random_device rd;
        std::uniform_int_distribution<> d(0, paramValues.size() - 1);

        for (size_t i{0}; i < n; ++i)
        {
            std::vector<T> v;
            v.reserve(m);

            for (size_t j{0}; j < m; ++j)
                v.emplace_back(paramValues[d(rd)]);

            values.emplace_back(v);
        }

        return values;
    }

    template <typename T>
    std::vector<T> roundParams(std::vector<T> const& params, std::vector<T> const& paramValues)
    {
        if (paramValues.empty())
            return params;

        std::vector<T> rounded;

        rounded.reserve(params.size());

        for (auto const& p : params)
        {
            auto bestVal = paramValues[0];
            auto bestDist = std::abs(p - bestVal);

            for (const T& v : paramValues)
            {
                auto const dist = std::abs(p - v);

                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestVal = v;
                }
            }

            rounded.emplace_back(bestVal);
        }

        return rounded;
    }

    template <typename T>
    struct ScoredVector
    {
        std::vector<T> values;
        T distance;

        bool operator<(const ScoredVector<T>& other) const
        {
            return distance < other.distance;
        }
    };

    std::string expr(std::string const& s)
    {
        auto const ca{std::ranges::count(s, 'a')};
        auto const cb{std::ranges::count(s, 'b')};
        auto e{s};
        size_t pos_a{0};
        size_t pos_b{0};

        for (int i{0}; i < ca; ++i)
        {
            pos_a = e.find('a', pos_a);
            e.replace(pos_a, 1, std::string("a") + std::to_string(i));
            pos_a += 1;
        }

        for (int i{0}; i < cb; ++i)
        {
            pos_b = e.find('b', pos_b);
            e.replace(pos_b, 1, std::string("b") + std::to_string(i));
            pos_b += 1;
        }

        return e;
    }

    size_t countOccurrences(std::string const& haystack, std::string const& needle)
    {
        if (needle.empty())
            return 0;

        size_t count = 0;
        size_t pos = 0;

        while ((pos = haystack.find(needle, pos)) != std::string::npos)
        {
            ++count;
            pos += needle.length();
        }

        return count;
    }

    std::string dotExpr(std::string const& s)
    {
        auto const ca{countOccurrences(s, "\"a\"")};
        auto const cb{countOccurrences(s, "\"b\"")};
        auto e{s};
        size_t pos_a{0};
        size_t pos_b{0};

        for (int i{0}; i < ca; ++i)
        {
            pos_a = e.find("\"a\"", pos_a);
            e.replace(pos_a, 3, std::string("\"a") + std::to_string(i) + "\"");
            pos_a += 1;
        }

        for (int i{0}; i < cb; ++i)
        {
            pos_b = e.find("\"b\"", pos_b);
            e.replace(pos_b, 3, std::string("\"b") + std::to_string(i) + "\"");
            pos_b += 1;
        }

        return e;
    }

    template <typename T>
    class Residual;

    template <typename T>
    class Expression
    {
        public:
            T a{1};
            T b{1};

            ceres::Jet<T, 4> ja;
            ceres::Jet<T, 4> jb;

            bool aFixed{false};
            bool bFixed{false};

            enum OperandType
            {
                NoOperand,
                VariableOperand,
                ExpressionOperand
            };

            enum OperatorType
            {
                LinearOp,
                UnaryOp,
                BinaryOp
            };

            Expression(Variable<T> const& variable) : operand1Type_{VariableOperand}, operand1Variable_{std::make_unique<Variable<T> >(variable)}, operand1Expression_{},
                                                      operand2Type_{NoOperand}, operand2Variable_{}, operand2Expression_{},
                                                      operatorType_{LinearOp}, unaryOperator_{}, binaryOperator_{}
            {
            }

            Expression(UnaryOperator<T> const& op, Expression const& operand)
             : operand1Type_{ExpressionOperand}, operand1Variable_{}, operand1Expression_{std::make_unique<Expression<T> >(operand)},
               operand2Type_{NoOperand}, operand2Variable_{}, operand2Expression_{},
               operatorType_{UnaryOp}, unaryOperator_{std::make_unique<UnaryOperator<T> >(op)}, binaryOperator_{}
            {
                for (auto const& v: operand.opTree_)
                    opTree_.emplace_back(v);

                opTree_.emplace_back(op.name());
            }

            Expression(BinaryOperator<T> const& op, Expression const& operand1, Expression const& operand2)
             : operand1Type_{ExpressionOperand}, operand1Variable_{}, operand1Expression_{std::make_unique<Expression<T> >(operand1)},
               operand2Type_{ExpressionOperand}, operand2Variable_{}, operand2Expression_{std::make_unique<Expression<T> >(operand2)},
               operatorType_{BinaryOp}, unaryOperator_{}, binaryOperator_{std::make_unique<BinaryOperator<T> >(op)}
            {
                for (auto const& v: operand1.opTree_)
                    opTree_.emplace_back(v);

                for (auto const& v: operand2.opTree_)
                    opTree_.emplace_back(v);

                opTree_.emplace_back(op.name());

                if (op.name() == "+" || op.name() == "-")
                {
                    aFixed = true;

                    operand1Expression_->bFixed = true;
                    operand2Expression_->bFixed = true;

                    if (*operand1Expression_ == *operand2Expression_)
                    {
                        operatorType_ = LinearOp;
                        operand2Type_ = NoOperand;
                        operand2Expression_.reset();
                    }
                }
                else if (op.name() == "*")
                {
                    auto op1_{operand1};
                    op1_.bFixed = true;

                    auto op2_{operand2};
                    op2_.bFixed = true;

                    auto op1{operand1};
                    op1.aFixed = true;
                    op1.bFixed = true;
                    operand1Expression_ = std::make_unique<Expression<T> >(op1);

                    auto op2{operand2};
                    op2.aFixed = true;
                    op2.bFixed = true;
                    operand2Expression_ = std::make_unique<Expression<T> >(op2);

                    bFixed = true;

                    Expression<T> const add{BinaryOperator<T>::plus(), *this,
                                            Expression<T>{BinaryOperator<T>::plus(), op1_, op2_}};

                    *this = std::move(add);
                }
            }

            Expression(Expression<T> const& other)
             : operand1Type_{other.operand1Type_}, operand1Variable_{}, operand1Expression_{},
               operand2Type_{other.operand2Type_}, operand2Variable_{}, operand2Expression_{},
               operatorType_{other.operatorType_}, unaryOperator_{}, binaryOperator_{},
               aFixed{other.aFixed}, bFixed{other.bFixed}, a{other.a}, b{other.b},
               ja{other.ja}, jb{other.jb}
            {
                if (other.operand1Variable_)
                    operand1Variable_ = std::make_unique<Variable<T> >(*other.operand1Variable_);

                if (other.operand1Expression_)
                    operand1Expression_ = std::make_unique<Expression<T> >(*other.operand1Expression_);

                if (other.operand2Variable_)
                    operand2Variable_ = std::make_unique<Variable<T> >(*other.operand2Variable_);

                if (other.operand2Expression_)
                    operand2Expression_ = std::make_unique<Expression<T> >(*other.operand2Expression_);

                if (other.unaryOperator_)
                    unaryOperator_ = std::make_unique<UnaryOperator<T> >(*other.unaryOperator_);

                if (other.binaryOperator_)
                    binaryOperator_ = std::make_unique<BinaryOperator<T> >(*other.binaryOperator_);
            }

            Expression& operator=(Expression<T> const& other)
            {
                operand1Type_ = other.operand1Type_;
                operand2Type_ = other.operand2Type_;
                operatorType_ = other.operatorType_;
                aFixed = other.aFixed;
                bFixed = other.bFixed;
                a = other.a;
                b = other.b;
                ja = other.ja;
                jb = other.jb;
                operand1Variable_.reset();
                operand1Expression_.reset();
                operand2Variable_.reset();
                operand2Expression_.reset();
                unaryOperator_.reset();
                binaryOperator_.reset();

                if (other.operand1Variable_)
                    operand1Variable_ = std::make_unique<Variable<T> >(*other.operand1Variable_);

                if (other.operand1Expression_)
                    operand1Expression_ = std::make_unique<Expression<T> >(*other.operand1Expression_);

                if (other.operand2Variable_)
                    operand2Variable_ = std::make_unique<Variable<T> >(*other.operand2Variable_);

                if (other.operand2Expression_)
                    operand2Expression_ = std::make_unique<Expression<T> >(*other.operand2Expression_);

                if (other.unaryOperator_)
                    unaryOperator_ = std::make_unique<UnaryOperator<T> >(*other.unaryOperator_);

                if (other.binaryOperator_)
                    binaryOperator_ = std::make_unique<BinaryOperator<T> >(*other.binaryOperator_);

                return *this;
            }

            bool operator!=(Expression<T> const& other) const
            {
                return !operator==(other);
            }

            bool operator==(Expression<T> const& other) const
            {
                if (operatorType_ != other.operatorType_)
                    return false;

                if (operatorType_ == BinaryOp && binaryOperator_->symmetry == BinaryOperator<T>::NonStrictSymmetry)
                {
                    if (*operand1Expression_ == *other.operand2Expression_ && *operand2Expression_ == *other.operand1Expression_)
                        return true;
                }

                if (operand1Type_ != other.operand1Type_)
                    return false;

                if (operand1Type_ == VariableOperand && *operand1Variable_ != *other.operand1Variable_)
                    return false;

                if (operand1Type_ == ExpressionOperand && *operand1Expression_ != *other.operand1Expression_)
                    return false;

                if (operand2Type_ != other.operand2Type_)
                    return false;

                if (operand2Type_ == VariableOperand && *operand2Variable_ != *other.operand2Variable_)
                    return false;

                if (operand2Type_ == ExpressionOperand && *operand2Expression_ != *other.operand2Expression_)
                    return false;

                return true;
            }

            Eigen::Array<T, Eigen::Dynamic, 1> eval() const
            {
                auto a_{a};
                auto b_{b};

                if (aFixed)
                    a_ = 1;

                if (bFixed)
                    b_ = 0;

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        return a_ * operand1Variable_->value() + b_;
                    else
                        return a_ * operand1Expression_->eval() + b_;
                }
                else if (operatorType_ == UnaryOp)
                    return a_ * unaryOperator_->op()(operand1Expression_->eval()) + b_;
                else// if (operatorType_ == BinaryOp)
                    return a_ * binaryOperator_->op()(operand1Expression_->eval(), operand2Expression_->eval()) + b_;
            }

            Eigen::Array<ceres::Jet<T, 4>, Eigen::Dynamic, 1> evalJets() const
            {
                auto ja_{ja};
                auto jb_{jb};

                if (aFixed)
                    ja_ = ceres::Jet<T, 4>{1};

                if (bFixed)
                    jb_ = ceres::Jet<T, 4>{0};

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        return ja_ * operand1Variable_->value() + jb_;
                    else
                        return ja_ * operand1Expression_->evalJets() + jb_;
                }
                else if (operatorType_ == UnaryOp)
                    return ja_ * unaryOperator_->jetOp()(operand1Expression_->evalJets()) + jb_;
                else// if (operatorType_ == BinaryOp)
                    return ja_ * binaryOperator_->jetOp()(operand1Expression_->evalJets(), operand2Expression_->evalJets()) + jb_;
            }

            std::string str() const
            {
                std::string s;

                if (!aFixed)
                    s = "a*(";

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        s += operand1Variable_->name();
                    else
                        s += operand1Expression_->str();
                }
                else if (operatorType_ == UnaryOp)
                    s += unaryOperator_->name() + "(" + operand1Expression_->str() + ")";
                else// if (operatorType_ == BinaryOp)
                {
                    auto const n{binaryOperator_->name()};

                    if (isSymbol(n))
                        s += "(" + operand1Expression_->str() + ")" + n + "(" + operand2Expression_->str() + ")";
                    else
                        s += n + "(" + operand1Expression_->str() + "," + operand2Expression_->str() + ")";
                }

                if (!aFixed)
                    s += ")";

                if (!bFixed)
                    s += "+b";

                return s;
            }

            std::string optStr() const
            {
                auto a_{a};
                auto b_{b};

                if (aFixed)
                    a_ = 1;

                if (bFixed)
                    b_ = 0;

                std::string s;

                if (std::abs(a_) > std::numeric_limits<T>::epsilon())
                {
                    if (!aFixed)
                        s += to_string_c_locale(a_) + "*(";

                    if (operatorType_ == LinearOp)
                    {
                        if (operand1Type_ == VariableOperand)
                            s += operand1Variable_->name();
                        else
                            s += operand1Expression_->optStr();
                    }
                    else if (operatorType_ == UnaryOp)
                        s += unaryOperator_->name() + "(" + operand1Expression_->optStr() + ")";
                    else// if (operatorType_ == BinaryOp)
                    {
                        auto const n{binaryOperator_->name()};

                        if (isSymbol(n))
                            s += "(" + operand1Expression_->optStr() + ")" + n + "(" + operand2Expression_->optStr() + ")";
                        else
                            s += n + "(" + operand1Expression_->optStr() + "," + operand2Expression_->optStr() + ")";
                    }

                    if (!aFixed)
                        s += ")";
                }

                if (std::abs(b_) > std::numeric_limits<T>::epsilon())
                {
                    if (!bFixed)
                    {
                        if (s.size())
                            s += "+";

                        s += to_string_c_locale(b_);
                    }
                }

                if (s.empty())
                    s = "0";

                return s;
            }

            std::string symStr() const
            {
                return symExpr().simplify().str();
            }

            sym::Expression<T> symExpr() const
            {
                auto a_{a};
                auto b_{b};

                if (aFixed)
                    a_ = 1;

                if (bFixed)
                    b_ = 0;

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        return a_ * sym::Symbol(operand1Variable_->name()) + b_;
                    else
                        return a_ * operand1Expression_->symExpr() + b_;
                }
                else if (operatorType_ == UnaryOp)
                    return a_ * unaryOperator_->symOp()(operand1Expression_->symExpr()) + b_;
                else// if (operatorType_ == BinaryOp)
                    return a_ * binaryOperator_->symOp()(operand1Expression_->symExpr(), operand2Expression_->symExpr()) + b_;
            }

            Expression<T> random() const
            {
                Expression<T> e{*this};

                if (e.operatorType_ == LinearOp)
                {
                    if (e.operand1Type_ == VariableOperand)
                    {
                        auto x{e.operand1Variable_->value()};
                        x.setRandom();
                        auto const y{2 * Eigen::Array<T, Eigen::Dynamic, 1>{x.abs()} * Eigen::Array<T, Eigen::Dynamic, 1>{e.operand1Variable_->value()}};
                        e.operand1Variable_ = std::make_unique<Variable<T> >(e.operand1Variable_->name(), y);
                    }
                    else
                        e.operand1Expression_ = std::make_unique<Expression<T> >(e.operand1Expression_->random());
                }
                else if (e.operatorType_ == UnaryOp)
                    e.operand1Expression_ = std::make_unique<Expression<T> >(e.operand1Expression_->random());
                else// if (e.operatorType_ == BinaryOp)
                {
                    e.operand1Expression_ = std::make_unique<Expression<T> >(e.operand1Expression_->random());
                    e.operand2Expression_ = std::make_unique<Expression<T> >(e.operand2Expression_->random());
                }

                return e;
            }

            std::string sympyStr(T eps = 1e-4) const
            {
                namespace py = pybind11;

                static py::scoped_interpreter guard{};

                try
                {
                    py::module sympy = py::module::import("sympy");

                    py::object expr = sympy.attr("sympify")(optStr());
                    py::object simplified = sympy.attr("simplify")(expr);
                    py::object expanded = sympy.attr("expand")(expr);

                    py::exec("from sympy import Float, Rational", py::globals());

                    std::string const eps_s{to_string_c_locale(eps)};
                    std::string s{"(lambda e: e.replace(lambda x: isinstance(x, (Float, Rational)), lambda x: Float(round(float(x)/" + eps_s + ") * " + eps_s + ")))"};

                    py::object round_expr = py::eval(s);

                    py::object expr_rounded = round_expr(expanded);
                    simplified = sympy.attr("simplify")(expr_rounded);

                    return py::str(simplified);
                }
                catch (py::error_already_set const& e)
                {
                    std::cout << e.what() << std::endl;
                }

                return optStr();
            }

            void params(std::vector<T>& params) const
            {
                if (!aFixed)
                    params.emplace_back(a);

                if (!bFixed)
                    params.emplace_back(b);

                if (operand1Type_ == ExpressionOperand)
                    operand1Expression_->params(params);

                if (operand2Type_ == ExpressionOperand)
                    operand2Expression_->params(params);
            }

            template <typename Container>
            void jets(Container& params) const
            {
                if (!aFixed)
                    params.emplace_back(ja);

                if (!bFixed)
                    params.emplace_back(jb);

                if (operand1Type_ == ExpressionOperand)
                    operand1Expression_->jets(params);

                if (operand2Type_ == ExpressionOperand)
                    operand2Expression_->jets(params);
            }

            std::vector<T> applyParams(std::vector<T> const& params)
            {
                size_t shift{0};

                if (!aFixed)
                {
                    a = params[shift];
                    ++shift;
                }

                if (!bFixed)
                {
                    b = params[shift];
                    ++shift;
                }

                std::vector<T> p{params.begin() + shift, params.end()};

                if (operand1Type_ == ExpressionOperand)
                    p = operand1Expression_->applyParams(p);

                if (operand2Type_ == ExpressionOperand)
                    p = operand2Expression_->applyParams(p);

                return p;
            }

            template <typename Container>
            Container applyJets(Container const& params)
            {
                size_t shift{0};

                if (!aFixed)
                {
                    ja = params[shift];
                    ++shift;
                }

                if (!bFixed)
                {
                    jb = params[shift];
                    ++shift;
                }

                Container p{params.begin() + shift, params.end()};

                if (operand1Type_ == ExpressionOperand)
                    p = operand1Expression_->applyJets(p);

                if (operand2Type_ == ExpressionOperand)
                    p = operand2Expression_->applyJets(p);

                return p;
            }

            T fit(Eigen::Array<T, Eigen::Dynamic, 1> const& y, std::vector<T> const& paramValues = std::vector<T>{}, T epsLoss = 1e-6, bool verbose = false, size_t exhaustiveLimit = 1e5, bool discreteParams = true, std::atomic<bool> const& timeoutTriggered = std::atomic<bool>{false})
            {
                std::vector<double> params;
                this->params(params);
                auto const possibilities{std::pow(paramValues.size(), params.size())};
                auto const n{params.size()};

                if (paramValues.size())
                {
                    std::set<Eigen::Array<T, Eigen::Dynamic, 1>, EigenArrayLess<T> > initialCells;

                    Eigen::Array<T, Eigen::Dynamic, 1> barycenter(params.size());

                    for (size_t i{0}; i < params.size(); ++i)
                        barycenter[i] = paramValues[paramValues.size() / 2];

                    initialCells.emplace(barycenter);

                    struct Cell
                    {
                        T cost;
                        Eigen::Array<T, Eigen::Dynamic, 1> param;
                        std::vector<Eigen::Array<T, Eigen::Dynamic, 1> > directions;
                        
                        Cell(T const& cost = T{0},
                             Eigen::Array<T, Eigen::Dynamic, 1> const& param = Eigen::Array<T, Eigen::Dynamic, 1>{},
                             std::vector<Eigen::Array<T, Eigen::Dynamic, 1> > const& directions = std::vector<Eigen::Array<T, Eigen::Dynamic, 1> >{})
                            : cost{cost}, param{param}, directions{directions}
                        {
                        }
                    };

                    std::set<Eigen::Array<T, Eigen::Dynamic, 1>, EigenArrayLess<T> > visitedCells;
                    std::vector<Cell> cells;

                    for (auto const& c: initialCells)
                    {
                        std::vector<Eigen::Array<T, Eigen::Dynamic, 1> > directions;

                        for (size_t i{0}; i < c.size(); ++i)
                        {
                            Eigen::Array<T, Eigen::Dynamic, 1> direction(c.size());
                            direction *= 0;
                            auto const j{static_cast<size_t>(std::distance(paramValues.begin(), std::find(paramValues.begin(), paramValues.end(), c[i])))};
                            direction[i] = paramValues[(j - 1) % paramValues.size()] - c[i];
                            directions.emplace_back(direction);
                            direction[i] = paramValues[(j + 1) % paramValues.size()] - c[i];
                            directions.emplace_back(direction);
                        }

                        std::vector<T> p;
                        p.reserve(c.size());

                        for (auto const& v: c)
                            p.emplace_back(v);

                        applyParams(p);

                        auto const x{eval()};
                        auto const cost{(y - x).square().sum()};

                        cells.emplace_back(cost, c, directions);
                    }

                    std::sort(cells.begin(), cells.end(), [] (auto const& x, auto const& y) {return x.cost < y.cost;});

                    auto bestCost{cells.front().cost};
                    auto bestParams{cells.front().param};

                    if (std::isnan(bestCost))
                        bestCost = std::numeric_limits<T>::infinity();

                    auto const limit{exhaustiveLimit};

                    while (cells.size())
                    {
                        if (timeoutTriggered)
                            break;

                        auto const cell{cells.front()};
                        cells.front() = cells.back();
                        cells.pop_back();

                        if (visitedCells.find(cell.param) == visitedCells.end())
                            visitedCells.emplace(cell.param);
                        else
                            continue;

                        if (visitedCells.size() > limit)
                            break;

                        for (auto const& direction: cell.directions)
                        {
                            if (timeoutTriggered)
                                break;

                            Eigen::Array<T, Eigen::Dynamic, 1> const c{cell.param + direction};

                            std::vector<Eigen::Array<T, Eigen::Dynamic, 1> > d;

                            for (size_t i{0}; i < c.size(); ++i)
                            {
                                Eigen::Array<T, Eigen::Dynamic, 1> direction(c.size());
                                direction *= 0;
                                auto const j{static_cast<size_t>(std::distance(paramValues.begin(), std::find(paramValues.begin(), paramValues.end(), c[i])))};
                                direction[i] = paramValues[(j - 1) % paramValues.size()] - c[i];
                                d.emplace_back(direction);
                                direction[i] = paramValues[(j + 1) % paramValues.size()] - c[i];
                                d.emplace_back(direction);
                            }

                            std::vector<T> p;
                            p.reserve(c.size());

                            for (auto const& v: c)
                                p.emplace_back(v);

                            applyParams(p);

                            auto const x{eval()};
                            auto const cost{(y - x).square().sum()};

                            if (cost < bestCost)
                            {
                                bestCost = cost;
                                bestParams = c;

                                if (cost < epsLoss)
                                    return cost;
                            }

                            cells.emplace_back(cost, c, d);
                        }

                        std::sort(cells.begin(), cells.end(), [] (auto const& x, auto const& y) {return x.cost < y.cost;});
                        cells.resize(50);
                    }

                    if (discreteParams)
                    {
                        std::vector<T> p;
                        p.reserve(bestParams.size());

                        for (auto const& v: bestParams)
                            p.emplace_back(v);

                        applyParams(p);

                        return bestCost;
                    }
                }

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
                options.minimizer_progress_to_stdout = verbose;
                options.logging_type = ceres::SILENT;

                ceres::Solver::Summary summary;
                ceres::Solve(options, &problem, &summary);

                applyParams(params);

                if (discreteParams)
                {
                    auto const roundedParams{roundParams(params, paramValues)};

                    applyParams(roundedParams);
                }

                auto const x{eval()};
                auto const loss{(y - x).square().sum()};

                return loss;
            }

            std::vector<std::string> const& opTree() const
            {
                return opTree_;
            }

            bool isNull(T eps = 1e-4) const
            {/*
                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                        return std::abs(a) < eps && std::abs(b) < eps;
                    else
                        return operand1Expression_->isNull() && (std::abs(b) < eps || bFixed);
                }
                else if (operatorType_ == UnaryOp)
                    return (std::abs(a) < eps || operand1Expression_->isNull()) && (std::abs(b) < eps || bFixed);
                else// if (operatorType_ == BinaryOp)
                {
                    if (binaryOperator_->name() == "+" || binaryOperator_->name() == "-")
                        return (std::abs(a) < eps || (operand1Expression_->isNull() && operand2Expression_->isNull())) && (std::abs(b) < eps || bFixed);
                    else if (binaryOperator_->name() == "*")
                        return (std::abs(a) < eps || (operand1Expression_->isNull() || operand2Expression_->isNull())) && (std::abs(b) < eps || bFixed);
                    else
                        return false;
                }*/
                auto const e{random()};
                Eigen::Matrix<T, Eigen::Dynamic, 1> const x{e.eval()};

                return (x.norm() < eps);
            }

            void updateNode(Node* node) const
            {
                std::string const varColor{"blue"};
                std::string const unColor{"red"};
                std::string const binColor{"purple"};
                std::string const aColor{"green"};
                std::string const bColor{"orange"};

                if (operatorType_ == LinearOp)
                {
                    if (operand1Type_ == VariableOperand)
                    {
                        if (!aFixed)
                        {
                            node->label = "*";
                            node->color = binColor;

                            node->children.emplace_back(new Node("a", aColor, to_string_c_locale(a)));
                            node->children.emplace_back(new Node(operand1Variable_->name(), varColor));
                        }
                        else
                            node->label = operand1Variable_->name();
                    }
                    else
                    {
                        if (!aFixed)
                        {
                            node->label = "*";
                            node->color = binColor;

                            node->children.emplace_back(new Node("a", aColor, to_string_c_locale(a)));
                            Node* n = new Node;
                            operand1Expression_->updateNode(n);
                            node->children.emplace_back(n);
                        }
                        else
                            operand1Expression_->updateNode(node);
                    }
                }
                else if (operatorType_ == UnaryOp)
                {
                    if (!aFixed)
                    {
                        node->label = "*";
                        node->color = binColor;

                        node->children.emplace_back(new Node("a", aColor, to_string_c_locale(a)));
                        Node* n1 = new Node(unaryOperator_->name(), unColor);
                        Node* n2 = new Node;
                        operand1Expression_->updateNode(n2);
                        n1->children.emplace_back(n2);
                        node->children.emplace_back(n1);
                    }
                    else
                    {
                        node->label = unaryOperator_->name();
                        node->color = unColor;

                        Node* n = new Node;
                        operand1Expression_->updateNode(n);
                        node->children.emplace_back(n);
                    }
                }
                else// if (operatorType_ == BinaryOp)
                {
                    if (!aFixed)
                    {
                        node->label = "*";
                        node->color = binColor;

                        node->children.emplace_back(new Node("a", aColor, to_string_c_locale(a)));
                        Node* n1 = new Node(binaryOperator_->name(), binColor);
                        Node* n2 = new Node;
                        operand1Expression_->updateNode(n2);
                        Node* n3 = new Node;
                        operand2Expression_->updateNode(n3);
                        n1->children.emplace_back(n2);
                        n1->children.emplace_back(n3);
                        node->children.emplace_back(n1);
                    }
                    else
                    {
                        node->label = binaryOperator_->name();
                        node->color = binColor;

                        Node* n1 = new Node;
                        operand1Expression_->updateNode(n1);
                        Node* n2 = new Node;
                        operand2Expression_->updateNode(n2);
                        node->children.emplace_back(n1);
                        node->children.emplace_back(n2);
                    }
                }

                if (!bFixed)
                {
                    if (node->label.empty())
                    {
                        node->label = "b";
                        node->color = bColor;
                        node->tooltip = to_string_c_locale(b);
                    }
                    else
                    {
                        auto const n{*node};
                        node->label = "+";
                        node->color = binColor;

                        node->children.clear();
                        node->children.emplace_back(new Node(n));
                        node->children.emplace_back(new Node("b", bColor, to_string_c_locale(b)));
                    }
                }
                else
                {
                    if (aFixed && node->label.empty())
                    {
                        node->label = "0";
                        node->color = "white";
                    }
                }
            }

            std::string dot(bool withSympyStr = true) const
            {
                std::ostringstream oss;

                oss << "digraph ExpressionTree {\n"
                    << "    node [shape=circle, style=filled, fillcolor=lightgray];\n"
                    << "\n";

                Node root;
                updateNode(&root);

                if (!root.tooltip.empty())
                    root.tooltip += "\n";

                root.tooltip += "Symbolic expression: " + expr(str()) + "\n"
                                + "Optimal expression: " + optStr();
                
                if (withSympyStr)
                     root.tooltip += "\nSympy expression: " + sympyStr();

                size_t counter{0};
                writeDot(&root, oss, counter);
                freeDot(&root);

                oss << "}\n";

                return dotExpr(oss.str());
            }

        private:
            OperandType operand1Type_;
            std::unique_ptr<Variable<T> > operand1Variable_;
            std::unique_ptr<Expression<T> > operand1Expression_;
            OperandType operand2Type_;
            std::unique_ptr<Variable<T> > operand2Variable_;
            std::unique_ptr<Expression<T> > operand2Expression_;
            OperatorType operatorType_;
            std::unique_ptr<UnaryOperator<T> > unaryOperator_;
            std::unique_ptr<BinaryOperator<T> > binaryOperator_;
            std::vector<std::string> opTree_;
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
    struct is_ceres_jet : std::false_type {};

    template <typename T, int N>
    struct is_ceres_jet<ceres::Jet<T, N>> : std::true_type {};

    template <typename T>
    struct Residual
    {
        Residual(Expression<T> const& expression,
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

            if constexpr (!is_ceres_jet<S>::value)
            {
                for (size_t i{0}; i < params.size(); ++i)
                    params[i] = *p[i];

                expression_.applyParams(params);

                auto const x{expression_.eval()};

                for (int i{0}; i < x.size(); ++i)
                {
                    if (ceres::isnan(x[i]))
                        return false;

                    residual[i] = S(y_[i]) - S(x[i]);
                }
            }
            else
            {
                std::vector<S> jets;
                expression_.jets(jets);

                for (size_t i{0}; i < jets.size(); ++i)
                    jets[i] = *p[i];

                expression_.applyJets(jets);

                auto const x{expression_.evalJets()};

                for (int i{0}; i < x.size(); ++i)
                {
                    if (ceres::isnan(x[i]))
                        return false;

                    residual[i] = S(y_[i]) - x[i];
                }
            }

            return true;
        }

        private:
            Expression<T> expression_;
            const Eigen::Array<double, Eigen::Dynamic, 1> y_;
    };
}

#endif // SYMREG_EXPRESSION_H
