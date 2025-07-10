#include "SymReg/SymbolicRegressor.h"

#include <QApplication>
#include <QBuffer>
#include <QDir>
#include <QLabel>
#include <QPixmap>
#include <QProcess>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QValueAxis>
#include <QSvgRenderer>
#include <QTemporaryFile>
#include <QTextStream>
#include <QThread>
#include <QTimer>
#include <QUrl>
#include <QWebEngineView>

extern "C"
{
    #include <graphviz/gvc.h>
    #include <graphviz/cgraph.h>
}

using namespace QtCharts;

using namespace sr;

QByteArray generateGraphvizSvg(QString const& dotSrc)
{
    GVC_t* gvc = gvContext();
    Agraph_t* g = agmemread(dotSrc.toUtf8().constData());

    if (!g)
        return {};

    gvLayout(gvc, g, "dot");

    char* data = nullptr;
    unsigned int length = 0;

    gvRenderData(gvc, g, "svg", &data, &length);

    QByteArray svgData(data, length);

    gvFreeRenderData(data);
    gvFreeLayout(gvc, g);
    agclose(g);
    gvFreeContext(gvc);

    return svgData;
}

QString svgToDataUrl(QByteArray const& svgData)
{
    QByteArray base64 = svgData.toBase64();

    return "data:image/svg+xml;base64," + base64;
}

QPixmap renderSvgToPixmap(QByteArray const& svgData, QSize const& size)
{
    if (svgData.isNull())
        return QPixmap();

    QSvgRenderer renderer(svgData);
    QPixmap pixmap(size);
    pixmap.fill(Qt::transparent);
    QPainter painter(&pixmap);
    renderer.render(&painter);

    return pixmap;
}

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
            
            std::cout << "Optimization end" << std::endl;
        }

        void setTargetY(Eigen::ArrayXd const& y_)
        {
            y = y_;
        }

    private:
        SymbolicRegressor<double>* sr;
        Eigen::ArrayXd y;
};

class ChartWithTooltip : public QChartView
{
    Q_OBJECT

    public:
        ChartWithTooltip(QChart* chart, QWidget* parent = nullptr)
            : QChartView(chart, parent), tooltip(new QGraphicsSimpleTextItem(chart))
        {
            setMouseTracking(true);
        }

    public slots:
        void showPointTooltip(QPointF point, bool state)
        {
            if (state)
            {
                QString text = QString("x = %1\ny = %2")
                                   .arg(point.x(), 0, 'f', 2)
                                   .arg(point.y(), 0, 'f', 2);
                tooltip->setText(text);
                QPointF pos = chart()->mapToPosition(point);
                tooltip->setPos(pos + QPointF(10, -30));
                tooltip->show();
            }
            else
                tooltip->hide();
        }

    private:
        QGraphicsSimpleTextItem *tooltip;
};

class DynamicChart : public QObject
{
    Q_OBJECT

    public:
        double bestLoss = std::numeric_limits<double>::infinity();

        DynamicChart(QChart *chart, ChartWithTooltip *chartView) : chart(chart), chartView(chartView), x{Eigen::ArrayXd::LinSpaced(100, 2, 101)}
        {
        }

    public slots:
        void callback(Expression<double> const& e, double const& loss)
        {
            if (loss < bestLoss)
            {
                bestLoss = loss;

                auto const str{e.sympyStr()};

                std::cout << loss << " " << str << std::endl;

                auto const y{e.eval()};

                QLineSeries *series = new QLineSeries();
                series->setName(QString::fromStdString(str));

                for (int i{0}; i < x.size(); ++i)
                    series->append(x[i], y[i]);

                chart->addSeries(series);

                QObject::connect(series, &QLineSeries::hovered,
                                 chartView, &ChartWithTooltip::showPointTooltip);
            }

            auto const svgData = generateGraphvizSvg(QString::fromStdString(e.dot()));
            
            if (!svgData.isNull())
            {
                view->setHtml("<html><body>" + svgData + "</body></html>");
                view->show();
            }
        }

        void start()
        {
            std::cout << "Optimization begin" << std::endl;

            Eigen::ArrayXd y(100);
            y << 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547;

            QLineSeries *series = new QLineSeries();
            series->setName("p(n)");

            for (int i{0}; i < x.size(); ++i)
                series->append(x[i], y[i]);

            chart->addSeries(series);

            QObject::connect(series, &QLineSeries::hovered,
                             chartView, &ChartWithTooltip::showPointTooltip);

            chart->createDefaultAxes();

            for (auto const& a: chart->axes())
            {
                if (auto* axis = qobject_cast<QValueAxis*>(a))
                {
                    axis->setGridLineVisible(true);
                    axis->setMinorTickCount(4);
                    axis->setMinorGridLineVisible(true);
                    axis->setLabelFormat("%d");
                }
            }

            using Var = Variable<double>;
            using UnOp = UnaryOperator<double>;
            using BinOp = BinaryOperator<double>;

            std::vector<double> const paramsValue{-1, 0, 1};

            std::map<std::string, size_t> operatorDepth;
            operatorDepth["log"] = 2;
            
            view = new QWebEngineView;
            view->resize(800, 600);

            srPtr = std::make_unique<SymbolicRegressor<double> >(std::vector<Var>{Var("n", x)},
                                                                 std::vector<UnOp>{UnOp::log()},
                                                                 std::vector<BinOp>{BinOp::plus(), BinOp::times()},
                                                                 2,
                                                                 paramsValue,
                                                                 operatorDepth,
                                                                 std::vector<Expression<double> >{},
                                                                 false,
                                                                 [this] (Expression<double> const& e, double const& loss)
                                                                 {
                                                                     QMetaObject::invokeMethod(this, [this, e, loss] () { this->callback(e, loss); }, Qt::QueuedConnection);
                                                                 },
                                                                 true/*false*/);

            worker = new FitWorker(srPtr.get());
            worker->setTargetY(y);
            thread = new QThread();

            worker->moveToThread(thread);

            connect(thread, &QThread::started, worker, &FitWorker::run);

            thread->start();
    }

    private:
        QChart* chart;
        ChartWithTooltip* chartView;
        Eigen::ArrayXd x;
        std::unique_ptr<SymbolicRegressor<double> > srPtr;
        FitWorker* worker;
        QThread* thread;
        QWebEngineView* view;
};


int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    QChart* chart = new QChart();
    chart->createDefaultAxes();
    chart->setTitle("Primes chart");

    ChartWithTooltip* chartView = new ChartWithTooltip(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    chartView->resize(600, 400);
    chartView->show();

    DynamicChart dynamicChart(chart, chartView);

    QTimer::singleShot(0, &dynamicChart, &DynamicChart::start);

    return app.exec();
}

#include "primes_demo.moc"
