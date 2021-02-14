#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <random>
#include <vector>
#include <cmath>

#include <iostream>

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;

std::default_random_engine gen;

const double PI  = 3.141592653589793238463;
const double PI2  = 2*PI;

struct FastMap{
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
        // Line() = default;
        Line(double a_, double b_, double c_): a(a_), b(b_), c(c_) {}; 
        // Line(const Line &) = default;
        // virtual ~Line() = default;

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
            // return true;
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

    vector<vector<bool>> get_grid(){
        vector<vector<bool>> grid(width);
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

    bool is_valid(double x, double y){
        bool ret = true;
        for (auto& o:objs){
            ret = ret && o->is_valid(x,y);
            if(!ret) break;
        }
        return ret;
    }

    void set_random_pos(double &x, double &y){
        static std::uniform_real_distribution<double> w(0.0,(double)width), // TODO 100 bo hack
                                                h(0.0,(double)height);
        do{
            x = w(gen);
            y = h(gen);
        }while(!is_valid(x,y));
    }
};

struct Model{
    struct State{
        double x,y,ori;
    } real_state;

    struct Measurment{
        double dist;
    };

    double vel;

    FastMap* map;

    Model(): real_state({0,0,0}), vel(0){}
    Model(double x_, double y_, double z_, double vel_): real_state({x_, y_, z_}), vel(vel_){}

    Measurment _get_meas(const State &st){
        Measurment ret;
        ret.dist = map->get_meas(st.x,st.y,st.ori);
        return ret;
    }

    State get_random_state(){
        static std::uniform_real_distribution<double> o(0.0,PI2);
        State st;
        map->set_random_pos(st.x,st.y);
        st.ori = o(gen);
        return st;
    }

    Measurment get_meas(){
        return _get_meas(real_state);
    }

    double get_meas_prob(const State &st, const Measurment &meas){
        double m = _get_meas(st).dist;
        return (1.-(double)fabs(meas.dist-m)/1000/sqrt(2.));//TODO hack 1000 bo taka mapa
    }

    void set_map(FastMap *mapc){
        map = mapc;
    }

    void set(double x, double y, double ori, double vel_){
        // py::print(__func__,"flush"_a=true);
        real_state.x = x;
        real_state.y = y;
        real_state.ori = ori;
        vel = vel_;
    }

    py::array get(){
        // py::print(__func__,"flush"_a=true);
        std::vector<double> ret{real_state.x,real_state.y,real_state.ori,vel};
        return py::cast(ret);
    }

    void update(double dori, double dvel){
        drift(real_state,dori,.0,.0);
    }

    void drift(State &p, double dori, double sigv, double sigori){
        static std::normal_distribution<double> dvn(0,sigv), dorin(0,sigori);
        double tvel = vel + dvn(gen);
        p.ori += dori + dorin(gen);
        if (map->get_meas(p.x,p.y,p.ori)<tvel) p.ori += PI;
        p.x += cos(p.ori)*tvel;
        p.y += sin(p.ori)*tvel;
    }

    bool is_valid(const State& st){
        return map->is_valid(st.x,st.y);
    }
};


enum RESAMPLE_TYPE{
    ROULETTE_WHEEL,
    SUS
};

struct ParticleFilter{
    Model* model;
    
    std::vector<Model::State> pop;
    std::vector<double> weights;

    py::array get_pop(){
        // py::print(__func__,"flush"_a=true);
        vector<vector<double>> ret;
        for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori});
        return py::cast(ret);
    }
    

    py::array get_weights(){
        // py::print(__func__,"flush"_a=true);
        return py::cast(weights);
    }

    void set_model(Model &model_){
        model=&model_;
    }

    void update_weights(Model::Measurment meas){
        // py::print(__func__,"flush"_a=true);
        weights.resize(pop.size());

        double sum=0;
        for (size_t i = 0;i<pop.size();++i){
            if(!model->is_valid(pop[i])){
                weights[i]=0;
                continue;
            }
            // double m = model._get_meas(pop[i]).dist;
            // m = model._get_meas(pop[i]).dist;
            // weights[i] = weights[i]*(1.-(double)fabs(meas-m)/1000/sqrt(2.));//TODO hack 100 bo taka mapa
            weights[i] = weights[i]*model->get_meas_prob(pop[i],meas);
            sum+=weights[i];
        }
        for(auto& w : weights) w/=sum;
    }

    double get_effective_N(){
        // py::print(__func__,"flush"_a=true);
        double sum = 0;
        for (auto& w:weights) sum+=w*w;
        return 1./sum;
    }

    void setup(size_t N){
        py::print("clearing pop");
        pop.resize(0);
        for (size_t i=0;i<N;++i){
            // Model::State st{w(gen),h(gen),o(gen)};
            // if(model.is_valid(st))
                // pop.push_back(st);
            pop.push_back(model->get_random_state());
            // else N++;
        }
        py::print("initialized pop",N);
        
        weights.resize(pop.size());
        for (auto & w : weights) w = 1./N;
        py::print("initialized weights");
    }

    py::tuple get_est(){
        // py::print(__func__,"flush"_a=true);
        double x=0,y=0,orix=0,oriy=0;
        for(size_t i = 0; i<pop.size();++i){
            orix+=cos(pop[i].ori)*weights[i];
            oriy+=sin(pop[i].ori)*weights[i];
            x+=pop[i].x*weights[i];
            y+=pop[i].y*weights[i];
        }
        double sum = 0;
        for (const auto& w: weights) sum+=w;
        x /= sum;
        y /= sum;
        orix /= sum;
        oriy /= sum;
        return py::make_tuple(x,y,atan2(oriy,orix));
    }
    
    void resample(RESAMPLE_TYPE type){
        // py::print(__func__,"flush"_a=true);
        std::vector<Model::State> new_pop;
        if (type == ROULETTE_WHEEL){
            std::discrete_distribution<int> dist(weights.begin(),weights.end());
            for (size_t i=0;i<pop.size();++i) new_pop.push_back(pop[dist(gen)]);
        }
        else{
            double sum = 0, wsum=0;
            for (const auto& e:weights) sum+=e;
            double step = sum/weights.size();
            double init = std::uniform_real_distribution<double>(0.,step)(gen);
            size_t j = 0;
            for( size_t i=0;i<pop.size();++i){
                double lw = init+step*i;
                while(wsum<lw){
                    wsum+=weights[j];
                    j++;
                }
                new_pop.push_back(pop[j]);
            }
        }
        pop = new_pop;
        for (auto& w: weights) w=1./pop.size();
    }

    void drift(double dori, double sigv, double sigori){
        // py::print(__func__,"flush"_a=true);
        for(auto &p : pop){
            model->drift(p, dori, sigv, sigori);
        }
    }
};


PYBIND11_MODULE(PFlib, m){
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::enum_<RESAMPLE_TYPE>(m, "RESAMPLE_TYPE")
        .value("ROULETTE_WHEEL",RESAMPLE_TYPE::ROULETTE_WHEEL)
        .value("SUS",RESAMPLE_TYPE::SUS).export_values();

    py::class_<ParticleFilter>(m,"ParticleFilter")
        .def(py::init<>())
        .def("get_effective_N", &ParticleFilter::get_effective_N)
        .def("get_est",&ParticleFilter::get_est)
        .def("setup", &ParticleFilter::setup)
        .def("get_pop", &ParticleFilter::get_pop)
        .def("get_weights", &ParticleFilter::get_weights)
        .def("set_model", &ParticleFilter::set_model)
        .def("update_weights", &ParticleFilter::update_weights)
        .def("resample", &ParticleFilter::resample)
        .def("drift", &ParticleFilter::drift);

    py::class_<FastMap>(m, "FastMap")
        .def(py::init<double>())
        .def("get_grid", &FastMap::get_grid)
        .def("add_line", &FastMap::add_line)
        .def("add_circle", &FastMap::add_circle)
        .def("get_meas", &FastMap::get_meas);

    py::class_<Model>(m, "Model")
        .def(py::init<double, double, double, double>())
        .def("set", &Model::set)
        .def("get", &Model::get)
        .def("get_meas", &Model::get_meas)
        .def("set_map", &Model::set_map)
        .def("update", &Model::update);

    py::class_<Model::Measurment>(m, "Measurment");
}
