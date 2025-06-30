#include <QApplication>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QThread>
#include <QTimer>

using namespace QtCharts;

#include "SymReg/SymbolicRegressor.h"

using namespace sr;

class FitWorker : public QObject
{
    Q_OBJECT

    public:
        FitWorker(SymbolicRegressor<double>* regressor) : sr(regressor)
        {
        }

    public slots:
        void run()
        {
            auto const result = sr->fit(y);
            emit finished(result.first, result.second);
        }

        void setTargetY(Eigen::ArrayXd const& y_)
        {
            y = y_;
        }

    signals:
        void finished(double loss, Expression<double> expr);

    private:
        SymbolicRegressor<double>* sr;
        Eigen::ArrayXd y;
};

class DynamicChart : public QObject
{
    Q_OBJECT

    public:
        double bestLoss = std::numeric_limits<double>::infinity();

        DynamicChart(QChart *chart) : chart(chart), x{Eigen::ArrayXd::LinSpaced(100, 2, 101)}
        {
        }

    public slots:
        void callback(Expression<double> const& e, double const& loss)
        {
            if (loss < bestLoss)
            {
                bestLoss = loss;
                
                std::cout << loss << " " << e.optStr() << std::endl;

                auto const y{e.eval()};

                QLineSeries *series = new QLineSeries();
                series->setName(QString::fromStdString(e.optStr()));
                
                for (int i{0}; i < x.size(); ++i)
                    series->append(x[i], y[i]);

                chart->addSeries(series);
            }
        }

        void start()
        {
            Eigen::ArrayXd y(100);
            y << 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547;

            QLineSeries *series = new QLineSeries();
            series->setName("p(n)");
            
            for (int i{0}; i < x.size(); ++i)
                series->append(x[i], y[i]);

            chart->addSeries(series);
            
            using Var = Variable<double>;
            using UnOp = UnaryOperator<double>;
            using BinOp = BinaryOperator<double>;

            std::vector<double> const paramsValue{-1, 0, 1};

            std::map<std::string, size_t> operatorDepth;
            operatorDepth["log"] = 2;
            operatorDepth["*"] = 4;
            operatorDepth["+"] = 4;

            srPtr = std::make_unique<SymbolicRegressor<double> >(std::vector<Var>{Var("n", x)},
                                                                 std::vector<UnOp>{UnOp::log()},
                                                                 std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                                                                 2,
                                                                 paramsValue,
                                                                 operatorDepth,
                                                                 std::vector<Expression<double>>{},
                                                                 false,
                                                                 [this](Expression<double> const& e, double const& loss)
                                                                 {
                                                                     QMetaObject::invokeMethod(this, [this, e, loss] () { this->callback(e, loss); }, Qt::QueuedConnection);
                                                                 });

            worker = new FitWorker(srPtr.get());
            worker->setTargetY(y);
            thread = new QThread();

            worker->moveToThread(thread);

            connect(thread, &QThread::started, worker, &FitWorker::run);
            connect(worker, &FitWorker::finished, this, [this](double loss, Expression<double> expr)
            {
                callback(expr, loss);
                thread->quit();
                worker->deleteLater();
                thread->deleteLater();
            });

        thread->start();
    }

    private:
        QChart* chart;
        Eigen::ArrayXd x;
        std::unique_ptr<SymbolicRegressor<double> > srPtr;
        FitWorker* worker;
        QThread* thread;
};


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    freopen("/tmp/stderr.txt", "w", stderr);

    QChart* chart = new QChart();
    chart->createDefaultAxes();
    chart->setTitle("Primes chart");

    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    chartView->resize(600, 400);
    chartView->show();
    
    DynamicChart dynamicChart(chart);

    QTimer::singleShot(0, &dynamicChart, &DynamicChart::start);

    return app.exec();
}

#include "primes_demo.moc"
