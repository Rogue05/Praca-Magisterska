#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <random>
#include <vector>
#include <memory>
#include <cmath>

#include <iostream>

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

const double PI  = 3.141592653589793238463;
const double PI2  = 2*PI;

struct PrimitiveMap{
private:
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

public:

    PrimitiveMap(double bound){
        width = bound;
        height = bound;
        add_line(-1.,0.,bound);
        add_line(0.,-1.,bound);
        add_line(1.,0.,0.);
        add_line(0.,1.,0.);
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

    double get_meas(double x, double y, double ori){
        double meas = 1000;
        for (auto& o:objs){
            auto m = o->get_dist(x,y,ori);
            if (m>0 && m<meas) meas = m;
        }
        return meas;
    }

    double get_meas_prob(double real, double x, double y, double ori){
        double meas = get_meas(x,y,ori);
        // double sig = (double)width*sqrt(2)/3;
        double sig = (double)width*sqrt(2)/30;
        // double sig = (double)width*sqrt(2)/18;
        double p = (meas-real)/sig;
        return exp(-p*p/2)/sig/sqrt(PI2);
        // return 2*(double)width-meas;
        // py::print("     ",x,meas,"        ",y,ori);
        // return meas;
    }

    bool is_valid(double x, double y){
        bool ret = true;
        for (auto& o:objs){
            ret = ret && o->is_valid(x,y);
            if(!ret) break;
        }
        return ret;
    }

    void set_random_pos(double &x, double &y){
        std::uniform_real_distribution<double> w(0.0,(double)width),
                                                h(0.0,(double)height);
        do{
            x = w(gen);
            y = h(gen);
        }while(!is_valid(x,y));
    }
};

struct robot_2d{
    double x, y, ori, vel;
    robot_2d(double x_, double y_, double ori_, double vel_)
        : x(x_), y(y_), ori(ori_), vel(vel_){}
    robot_2d(): robot_2d(0.0, 0.0, 0.0, 0.0){}
};

#define MAGIC_VELOCITY_MIN 5
#define MAGIC_VELOCITY_MAX 15

struct Model{
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand_ori, rand_vel;

    PrimitiveMap& map;

    Model(PrimitiveMap& map_, double do_, double dv_)
        :map(map_), rand_ori(-do_,do_), rand_vel(-dv_,dv_){};

    py::array_t<robot_2d> get_random_pop(size_t N){
        auto ret = py::array_t<robot_2d>(N);
        py::buffer_info buf = ret.request();
        auto ptr = static_cast<robot_2d*>(buf.ptr);
        
        std::uniform_real_distribution<double>
            ro(0.0,PI2), vo(MAGIC_VELOCITY_MIN, MAGIC_VELOCITY_MAX);
        for(size_t i=0; i < N; ++i){
            ptr[i] = {0.0, 0.0, ro(gen), vo(gen)};
            map.set_random_pos(ptr[i].x, ptr[i].y);
        }
        return ret;
    }

    py::array_t<robot_2d> get_linear_pop(size_t N){
        auto ret = py::array_t<robot_2d>(N);
        py::buffer_info buf = ret.request();
        auto ptr = static_cast<robot_2d*>(buf.ptr);
        
        for(size_t i=1; i < N-1; ++i){
            ptr[i] = robot_2d((double)i, (double)N/2, 0.0, 0.0);
            // py::print((double)i, ptr[i].x, ptr[i].y);
            // map.set_random_pos(ptr[i].x, ptr[i].y);
        }
        return ret;
    }

    void drift_state(robot_2d& state, double dori, double dvel){
        state.vel += dvel;
        state.ori += dori;
        if (map.get_meas(state.x,state.y,state.ori)<state.vel)
            state.ori += PI;
        state.x += cos(state.ori)*state.vel;
        state.y += sin(state.ori)*state.vel;
    }

    void drift(py::array_t<robot_2d> a_pop, double dori, double dvel){
        auto pop = a_pop.mutable_unchecked<1>();
        for(size_t i=0; i < pop.shape(0); ++i){
            pop(i).ori += rand_ori(gen);
            pop(i).vel += rand_vel(gen);
            drift_state(pop(i), dori, dvel);
        }
    }

    double get_meas(robot_2d& state){
        return map.get_meas(state.x,state.y,state.ori);
    }

    void update(py::array_t<robot_2d> a_pop, double dori, double dvel){
        auto pop = a_pop.mutable_unchecked<1>();
        for(size_t i=0; i < pop.shape(0); ++i){
            pop(i).ori += dori;
            pop(i).vel += dvel;
        }
    }

    // std::vector<double> update_weights(
    //     double real, std::vector<robot_2d> pop,
    //     std::vector<double> weights){
        
    //     double sum = 0.0;
    //     for(size_t i=0; i < pop.size(); ++i){
    //         weights[i] = weights[i]*map.get_meas_prob(real, pop[i].x, pop[i].y, pop[i].ori);
    //         sum += weights[i];
    //     }
    //     for(size_t i=0; i < pop.size(); ++i){
    //         weights[i] /= sum;
    //     }
    //     return weights;
    // }

    py::array_t<double> update_weights(
        double real, py::array_t<robot_2d> a_pop,
        py::array_t<double> a_weights){

        auto pop = a_pop.mutable_unchecked<1>();
        auto a_ret = py::array_t<double>(pop.shape(0));
        auto ret = a_ret.mutable_unchecked<1>();
        auto weights = a_weights.mutable_unchecked<1>();
        
        double sum = 0.0;
        for(size_t i=0; i < pop.shape(0); ++i){
            // py::print(i);
            ret(i) = weights(i)*map.get_meas_prob(real, pop(i).x, pop(i).y, pop(i).ori);
            sum += ret(i);
        }
        for(size_t i=0; i < pop.shape(0); ++i){
            ret(i) /= sum;
        }
        return a_ret;
    }

    py::array_t<double> get_weights(size_t N){
        auto a_ret = py::array_t<double>(N);
        auto ret = a_ret.mutable_unchecked<1>();
        
        const double w = 1.0/N;
        for(size_t i=0; i < ret.shape(0); ++i){
            ret(i) = w;
        }
        return a_ret;
    }

    py::array_t<double> as_array(py::array_t<robot_2d> a_pop){
        auto pop = a_pop.mutable_unchecked<1>();
        std::vector<std::vector<double>> ret(2);

        for(size_t i=0; i < pop.shape(0); ++i){
            ret[0].push_back(pop(i).x);
            ret[1].push_back(pop(i).y);
        }

        return py::cast(ret);
    }
};

py::array roulette_wheel_resample(py::array a_pop, py::array_t<double> a_weights){
    py::array a_new_pop = a_pop;
    py::buffer_info buf = a_pop.request();
    py::buffer_info new_buf = a_new_pop.request();

    auto weights = a_weights.mutable_unchecked<1>();

    std::default_random_engine gen;
    std::discrete_distribution<int> dist(weights.data(0),weights.data(0)+weights.shape(0));

    for(int i = 0; i < buf.shape[0]; ++i){
        std::memcpy(
            (char*)new_buf.ptr+i*buf.strides[0],
            (char*)buf.ptr+dist(gen)*buf.strides[0],
            buf.strides[0]);
    }

    return a_new_pop;
}

// std::vector<robot_2d> sus_resample(
//     std::vector<robot_2d> pop, std::vector<double> weights){
    
//     std::vector<robot_2d> new_pop(pop.size());

//     std::default_random_engine gen;
    
//     double sum = 0, wsum=weights[0];
//     for (size_t i = 0; i < weights.size(); ++i) sum+=weights[i];
//     double step = sum/weights.size();
//     double init = std::uniform_real_distribution<double>(0.,step)(gen);
//     size_t j = 0;

//     for (size_t i=0; i < pop.size(); ++i){
//         double lw = init+step*i;
//         while(wsum<lw){
//             j++;
//             wsum+=weights[j];
//         }
//         new_pop[i] = pop[j];
//     }

//     return new_pop;
// }

py::array sus_resample(py::array a_pop, py::array_t<double> a_weights){
    py::buffer_info buf = a_pop.request();

    py::array a_new_pop(pybind11::dtype(buf), buf.shape, buf.strides, nullptr);
    py::buffer_info new_buf = a_new_pop.request();

    auto weights = a_weights.mutable_unchecked<1>();

    std::default_random_engine gen;
    

    double sum = 0, wsum=weights(0);
    for (size_t i = 0; i < weights.shape(0); ++i) sum+=weights(i);
    double step = sum/weights.shape(0);
    double init = std::uniform_real_distribution<double>(0.,step)(gen);
    size_t j = 0;

    for (size_t i=0; i < buf.shape[0]; ++i){
        double lw = init+step*i;
        while(wsum<lw){
            j++;
            wsum+=weights(j);
        }
        std::memcpy(
            (char*)new_buf.ptr + i*buf.strides[0],
            (char*)buf.ptr + j*buf.strides[0],
            buf.strides[0]);
    }

    return a_new_pop;
}

PYBIND11_MODULE(PFlib, m){
    m.doc() = "particle filter lib";

    m.def("roulette_wheel_resample",roulette_wheel_resample);
    m.def("sus_resample",sus_resample);

    py::class_<robot_2d>(m,"robot_2d")
        .def(py::init<>())
        .def(py::init<double,double,double,double>())
        .def_readwrite("x", &robot_2d::x)
        .def_readwrite("y", &robot_2d::y)
        .def_readwrite("ori", &robot_2d::ori)
        .def_readwrite("vel", &robot_2d::vel);


    PYBIND11_NUMPY_DTYPE(robot_2d, x, y, ori, vel);


    py::class_<Model>(m, "Model")
        .def(py::init<PrimitiveMap&,double,double>())
        .def("get_random_pop", &Model::get_random_pop)
        .def("get_linear_pop", &Model::get_linear_pop)
        .def("drift_state",&Model::drift_state)
        .def("drift",&Model::drift)
        .def("get_meas",&Model::get_meas)
        .def("update_weights",&Model::update_weights)
        .def("get_weights",&Model::get_weights)
        .def("as_array",&Model::as_array);

    py::class_<PrimitiveMap>(m, "PrimitiveMap")
        .def(py::init<double>())
        .def("get_grid", &PrimitiveMap::get_grid)
        .def("add_line", &PrimitiveMap::add_line)
        .def("add_circle", &PrimitiveMap::add_circle);
}
