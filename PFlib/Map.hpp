#pragma once
#include <random>

#include <pybind11/pybind11.h> // print
#include <pybind11/numpy.h>

namespace py = pybind11;

const double PI  = 3.141592653589793238463;
const double PI2  = 2*PI;

struct Map{
    virtual double get_meas(double, double, double){
        py::print(__func__, "is not implemented");
        return 0.0;
    }
    virtual void set_random_pos(double&, double&){
        py::print(__func__, "is not implemented");
    }
    virtual bool is_valid(double, double){
        py::print(__func__, "is not implemented");
        return true;
    }
    virtual ~Map() = default;
};

struct FastMap : public Map{
// struct FastMap{
    std::default_random_engine gen;

    class CollObj{
    public:
        virtual double get_dist(double x, double y, double ori){
            py::print("Missing get_dist impl");
            return -1.0;
        }
        virtual bool is_valid(double x, double y){
            py::print("Missing is_valid impl");
            return false;
        }
        virtual ~CollObj() = default;
    };

    std::vector<std::unique_ptr<CollObj>> objs;
    int width,height;

    class Line: public CollObj{
        double a,b,c;
    public:
        Line(double a_, double b_, double c_): a(a_), b(b_), c(c_) {}; 

        double get_dist(double x, double y, double ori) override{
            double va=cos(ori), vb=-sin(ori), vc = -(va*y+vb*x);
            double M = -(a*vb-b*va);
            // py::print(M,a,vb,b,va);
            if (M*M<1.e-3) return -1;

            double mx = (-va*c+a*vc)/M-x;
            double my = (-b*vc+vb*c)/M-y;
            double m = sqrt(mx*mx + my*my);
            double ex = m*cos(ori)-mx,
                   ey = m*sin(ori)-my;

            if (ex*ex < 1.e-1 && ey*ey < 1.e-1) return m;
            return -1;
        }
        bool is_valid(double x, double y) override{
            return a*y+b*x+c > 0;
        }
    };

    class Circle: public CollObj{
        double cx,cy,r;
    public:
        Circle(double cx_, double cy_, double r_): cx(cx_), cy(cy_), r(r_) {}; 

        double get_dist(double x, double y, double ori) override{
            double va=cos(ori), vb=-sin(ori), vc = -(va*y+vb*x);
            double d = abs(va*cy+vb*cx+vc)/sqrt(va*va+vb*vb);
            if (d>r) return -1;
            double D = sqrt((cx-x)*(cx-x)+(cy-y)*(cy-y));
            double dx = sqrt(r*r-d*d);
            double m = sqrt(D*D-d*d)-dx;

            double ex = x + m*cos(ori), ey = y + m*sin(ori);
            double err = (ex-cx)*(ex-cx)+(ey-cy)*(ey-cy)-r*r;
            
            if(err*err<1.0e-1)
                return m;
            return -1;
        }
        bool is_valid(double x, double y){
            return (cx-x)*(cx-x)+(cy-y)*(cy-y) > r*r;
        }
    };

    FastMap(double bound){
        width = bound;
        height = bound;
        add_line(-1.,0.,bound);
        add_line(0.,-1.,bound);
        add_line(1.,0.,0.);
        add_line(0.,1.,0.);
    }

    static std::shared_ptr<FastMap> create(double bound){
        return std::make_shared<FastMap>(bound);
    }

    std::vector<std::vector<bool>> get_grid(){
        std::vector<std::vector<bool>> grid(width);
        for(auto& i:grid) i.resize(height);

        for(size_t x=0;x<width;++x){
            for(size_t y=0;y<height;++y){
                if(is_valid((double)x,
                            (double)y))
                    grid[x][y]= true;
            }
        }
        return grid;
    }

    void add_line(double a, double b, double c){
        objs.push_back(std::make_unique<Line>(a,b,c));
    }

    void add_circle(double cx, double cy, double r){
        objs.push_back(std::make_unique<Circle>(cx,cy,r));
    }

    double get_meas(double x, double y, double ori) override{
    // double get_meas(double x, double y, double ori){
        double meas = 1000;
        for (auto& o:objs){
            auto m = o->get_dist(x,y,ori);
            if (m>0 && m<meas) meas = m;
        }
        return meas;
    }

    bool is_valid(double x, double y) override{
    // bool is_valid(double x, double y){
        bool ret = true;
        for (auto& o:objs){
            ret = ret && o->is_valid(x,y);
            if(!ret) break;
        }
        return ret;
    }

    void set_random_pos(double &x, double &y) override{
    // void set_random_pos(double &x, double &y){
        std::uniform_real_distribution<double> w(0.0,(double)width), // TODO 100 bo hack
                                                h(0.0,(double)height);
        do{
            x = w(gen);
            y = h(gen);
        }while(!is_valid(x,y));
    }
};

// struct HeightMap : public Map{
//     std::vector<std::vector<double>> grid;
//     HeightMap(py::array_t<double> grid_){
//         auto r = grid_.unchecked<2>();
//         grid.resize(r.shape(0));
//         for (auto& g:grid)
//             g.resize(r.shape(1));
//         for (py::ssize_t x = 0; x < r.shape(0); ++x)
//             for (py::ssize_t y = 0; y < r.shape(1); ++y)
//                 grid[x][y] = r(x,y);
//         // py::print(r.shape(0),r.shape(1),r.shape(0)*r.shape(1));
//     }

//     double get_meas(double x, double y, double ori) override{
//         return 2.;
//     }

//     void set_random_pos(double& x, double& y) override{
//         return;
//     }

//     bool is_valid(double x, double y) override{
//         return true;
//     }
// };