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

// vector<vector<bool>> grid;
// vector<vector<vector<bool>>> tree;

struct State{
    double x,y,ori;
};

struct Model : State{
    double vel;
} model;

// std::vector<State> pop;
// std::vector<double> weights;
std::default_random_engine gen;

const double PI  = 3.141592653589793238463;
const double PI2  = 2*PI;

// double _get_meas(double x, double y, double ori){
//     int i = 0;
//     double dx = 1,ret=0;

//     while(dx>1e-3){
//         while(grid
//             [x+(cos(ori)*(dx+ret))]
//             [y+(sin(ori)*(dx+ret))]==false) {
//                 ret+=dx;
//             }
//         dx/=2;
//     }
//     return ret;
// }

// void init_pop(size_t N){
//     // py::print(__func__,"flush"_a=true);
//     std::uniform_real_distribution<double> w(0.0,grid.size()),
//                                             h(0.0,grid[0].size()),
//                                             o(0.0,PI2);
//     py::print("clearing pop");
//     pop.resize(0);
//     for (size_t i=0;i<N;++i){
//         pop.push_back({w(gen),h(gen),o(gen)});
//     }
//     py::print("initialized pop");
    
//     weights.resize(pop.size());
//     for (auto & w : weights) w = 1./N;
//     py::print("initialized weights");
// }

// py::array get_pop(){
//     // py::print(__func__,"flush"_a=true);
//     vector<vector<double>> ret;
//     for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori});
//     return py::cast(ret);
// }

// py::array get_weights(){
//     // py::print(__func__,"flush"_a=true);
//     return py::cast(weights);
// }

// void set_model(double x, double y, double ori, double vel){
//     // py::print(__func__,"flush"_a=true);
//     model.x = x;
//     model.y = y;
//     model.ori = ori;
//     model.vel = vel;
// }

// void update_weights(double meas){
//     // py::print(__func__,"flush"_a=true);
//     weights.resize(pop.size());

//     double sum=0;
//     for (size_t i = 0;i<pop.size();++i){
//         if (pop[i].x < 0 || pop[i].x > grid.size() ||
//             pop[i].y < 0 || pop[i].y > grid[0].size() ||
//             grid[(int)pop[i].x][(int)pop[i].y]){
//                 weights[i]=0;
//                 continue;
//             }
//         double m = _get_meas(pop[i].x,pop[i].y,pop[i].ori);

//         weights[i] = weights[i]*(1.-(double)fabs(meas-m)/grid.size()/sqrt(2.));
//         sum+=weights[i];
//     }
//     for(auto& w : weights) w/=sum;
// }

// double get_effective_N(){
//     // py::print(__func__,"flush"_a=true);
//     double sum = 0;
//     for (auto& w:weights) sum+=w*w;
//     return 1./sum;
// }

enum RESAMPLE_TYPE{
    ROULETTE_WHEEL,
    SUS
};

// void resample(RESAMPLE_TYPE type){
//     // py::print(__func__,"flush"_a=true);
//     std::vector<State> new_pop;
//     if (type == ROULETTE_WHEEL){
//         std::discrete_distribution<int> dist(weights.begin(),weights.end());
//         for (size_t i=0;i<pop.size();++i) new_pop.push_back(pop[dist(gen)]);
//     }
//     else{
//         double sum = 0, wsum=0;
//         for (const auto& e:weights) sum+=e;
//         double step = sum/weights.size();
//         double init = std::uniform_real_distribution<double>(0.,step)(gen);
//         size_t j = 0;
//         for( size_t i=0;i<pop.size();++i){
//             double lw = init+step*i;
//             while(wsum<lw){
//                 wsum+=weights[j];
//                 j++;
//             }
//             new_pop.push_back(pop[j]);
//         }
//     }
//     pop = new_pop;
//     for (auto& w: weights) w=1./pop.size();
// }

// void drift(double dori){
//     // py::print(__func__,"flush"_a=true);
//     for(auto &p : pop){
//         // py::print("elem",p.x,p.y,"flush"_a=true);
//         // py::print((int)p.x,(int)p.y);
//         // py::print((int)grid[(int)p.x][(int)p.y]);
//         if (p.x < 0 || p.x >= grid.size() ||
//             p.y < 0 || p.y >= grid[0].size() ||
//             grid[(int)p.x][(int)p.y]) continue;
//         // py::print("go","flush"_a=true);
//         p.ori += dori;
//         double tmpx = p.x + cos(p.ori)*model.vel;
//         double tmpy = p.y + sin(p.ori)*model.vel;
//         // if (grid[(int)tmpx][(int)tmpy]) p.ori += PI;
//         if (tmpx < 0 || tmpx >= grid.size()-1 ||
//             tmpy < 0 || tmpy >= grid[0].size()-1 ||
//             grid[(int)tmpx][(int)tmpy]) p.ori += PI;
//         p.x += cos(p.ori)*model.vel;
//         p.y += sin(p.ori)*model.vel;
//     }
// }

// py::tuple get_est(){
//     // py::print(__func__,"flush"_a=true);
//     double x=0,y=0,orix=0,oriy=0;
//     for(size_t i = 0; i<pop.size();++i){
//         orix+=cos(pop[i].ori)*weights[i];
//         oriy+=sin(pop[i].ori)*weights[i];
//         x+=pop[i].x*weights[i];
//         y+=pop[i].y*weights[i];
//     }
//     double sum = 0;
//     for (const auto& w: weights) sum+=w;
//     x /= sum;
//     y /= sum;
//     orix /= sum;
//     oriy /= sum;
//     return py::make_tuple(x,y,atan2(oriy,orix));
// }

// void diffuse(double sigp, double sigori){
//     // py::print(__func__,"flush"_a=true);
//     std::normal_distribution<double> dp(0,sigp), dori(0,sigori);
//     for(auto &p : pop){
//         p.ori += dori(gen);
//         p.x += dp(gen);
//         p.y += dp(gen);
//         if (p.x<0.) p.x=0.;
//         if (p.y<0.) p.y=0.;
//         if (p.x>=grid.size()) p.x=grid.size()-1;
//         if (p.y>=grid[0].size()) p.y=grid[0].size()-1;
//     }
// }

class Map{
    vector<vector<bool>> grid;
    public:
    void add_circle(py::tuple pos,float radius){
        py::print(__func__,"flush"_a=true);
        for(size_t x=0;x<grid.size();++x){
            for(size_t y=0;y<grid.size();++y){
                float x2=(pos[0].cast<float>()-x);
                float y2=(pos[1].cast<float>()-y);
                if(x2*x2+y2*y2 <= radius*radius)
                    grid[x][y] = true;
            }
        }
    }

    void add_box(py::tuple pos, py::tuple size){
        // py::print(__func__,"flush"_a=true);
        for (int x=pos[0].cast<int>();x<pos[0].cast<int>()+size[0].cast<int>();++x)
            for (int y=pos[1].cast<int>();y<pos[1].cast<int>()+size[1].cast<int>();++y)
                if (x>=0 && y>=0 && x<grid.size() && y<grid.size())
                    grid[x][y]=true;
    }

    py::array get(){
        return py::cast(grid);
    }
    
    vector<vector<bool>> get_raw(){
        return grid;
    }

    void setup(size_t x,size_t y,size_t margin) {
        // py::print(__func__,"flush"_a=true);
        grid.resize(x);
        for (size_t i =0;i<x;++i) grid[i].resize(y);
        
        for(size_t x=0;x<grid.size();++x){
            for(size_t y=0;y<grid[0].size();++y){
                if (x<margin || x>grid.size()-margin || y<margin || y>grid[0].size()-margin)
                    grid[x][y] = true;
                else grid[x][y]=false;
            }
        }
    }
    
};

struct ParticleFilter{
    vector<vector<bool>> grid;
    
    std::vector<State> pop;
    std::vector<double> weights;

    py::array get_pop(){
        // py::print(__func__,"flush"_a=true);
        vector<vector<double>> ret;
        for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori});
        return py::cast(ret);
    }
    
    double _get_meas(double x, double y, double ori){
        int i = 0;
        double dx = 1,ret=0;

        while(dx>1e-3){
            while(grid
                [x+(cos(ori)*(dx+ret))]
                [y+(sin(ori)*(dx+ret))]==false) {
                    ret+=dx;
                }
            dx/=2;
        }
        return ret;
    }

    py::array get_weights(){
        // py::print(__func__,"flush"_a=true);
        return py::cast(weights);
    }

    void set_model(double x, double y, double ori, double vel){
        // py::print(__func__,"flush"_a=true);
        model.x = x;
        model.y = y;
        model.ori = ori;
        model.vel = vel;
    }

    void update_weights(double meas){
        // py::print(__func__,"flush"_a=true);
        weights.resize(pop.size());

        double sum=0;
        for (size_t i = 0;i<pop.size();++i){
            if (pop[i].x < 0 || pop[i].x > grid.size() ||
                pop[i].y < 0 || pop[i].y > grid[0].size() ||
                grid[(int)pop[i].x][(int)pop[i].y]){
                    weights[i]=0;
                    continue;
                }
            double m = _get_meas(pop[i].x,pop[i].y,pop[i].ori);

            weights[i] = weights[i]*(1.-(double)fabs(meas-m)/grid.size()/sqrt(2.));
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

    void set_map(Map &mapc){
        grid = mapc.get_raw();
    }

    void setup(size_t N){
        // py::print(__func__,"flush"_a=true);
        std::uniform_real_distribution<double> w(0.0,grid.size()),
                                                h(0.0,grid[0].size()),
                                                o(0.0,PI2);
        py::print("clearing pop");
        pop.resize(0);
        for (size_t i=0;i<N;++i){
            pop.push_back({w(gen),h(gen),o(gen)});
        }
        py::print("initialized pop");
        
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
    
    void diffuse(double sigp, double sigori){
        // py::print(__func__,"flush"_a=true);
        std::normal_distribution<double> dp(0,sigp), dori(0,sigori);
        for(auto &p : pop){
            p.ori += dori(gen);
            p.x += dp(gen);
            p.y += dp(gen);
            if (p.x<0.) p.x=0.;
            if (p.y<0.) p.y=0.;
            if (p.x>=grid.size()) p.x=grid.size()-1;
            if (p.y>=grid[0].size()) p.y=grid[0].size()-1;
        }
    }
    
    void resample(RESAMPLE_TYPE type){
        // py::print(__func__,"flush"_a=true);
        std::vector<State> new_pop;
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

    void drift(double dori){
        // py::print(__func__,"flush"_a=true);
        for(auto &p : pop){
            // py::print("elem",p.x,p.y,"flush"_a=true);
            // py::print((int)p.x,(int)p.y);
            // py::print((int)grid[(int)p.x][(int)p.y]);
            if (p.x < 0 || p.x >= grid.size() ||
                p.y < 0 || p.y >= grid[0].size() ||
                grid[(int)p.x][(int)p.y]) continue;
            // py::print("go","flush"_a=true);
            p.ori += dori;
            double tmpx = p.x + cos(p.ori)*model.vel;
            double tmpy = p.y + sin(p.ori)*model.vel;
            // if (grid[(int)tmpx][(int)tmpy]) p.ori += PI;
            if (tmpx < 0 || tmpx >= grid.size()-1 ||
                tmpy < 0 || tmpy >= grid[0].size()-1 ||
                grid[(int)tmpx][(int)tmpy]) p.ori += PI;
            p.x += cos(p.ori)*model.vel;
            p.y += sin(p.ori)*model.vel;
        }
    }

    // void set_map(Map &map){
    //     grid = map.get_raw();
    // }
};


// void tmp_set_map(Map &map){
//     grid = map.get_raw();
// }


PYBIND11_MODULE(PFlib, m){
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::enum_<RESAMPLE_TYPE>(m, "RESAMPLE_TYPE")
        .value("ROULETTE_WHEEL",RESAMPLE_TYPE::ROULETTE_WHEEL)
        .value("SUS",RESAMPLE_TYPE::SUS).export_values();

    // m.def("_get_meas", &_get_meas, "PAss");
    // m.def("get_effective_N", &get_effective_N, "PAss");
    // m.def("get_est",&get_est, "PAss");
    // m.def("init_pop", &init_pop, "PAss");
    // m.def("get_pop", &get_pop, "PAss");
    // m.def("get_weights", &get_weights);
    // m.def("set_model", &set_model, "PAss");
    // m.def("update_weights", &update_weights, "PAss");
    // m.def("resample", &resample, "PAss");
    // m.def("drift", &drift, "PAss");
    // m.def("diffuse", &diffuse, "PAss");
    
    py::class_<ParticleFilter>(m,"ParticleFilter")
        .def(py::init<>())
        .def("_get_meas", &ParticleFilter::_get_meas)
        .def("get_effective_N", &ParticleFilter::get_effective_N)
        .def("get_est",&ParticleFilter::get_est)
        .def("setup", &ParticleFilter::setup)
        .def("get_pop", &ParticleFilter::get_pop)
        .def("get_weights", &ParticleFilter::get_weights)
        .def("set_model", &ParticleFilter::set_model)
        .def("update_weights", &ParticleFilter::update_weights)
        .def("resample", &ParticleFilter::resample)
        .def("drift", &ParticleFilter::drift)
        .def("set_map", &ParticleFilter::set_map)
        .def("diffuse", &ParticleFilter::diffuse);
    

    // m.def("add_box", &add_box, "PAss");
    // m.def("add_circle", &add_circle, "PAss");
    // m.def("setup_map", &setup_map, "PAss");
    // m.def("get_grid", &get_grid, "PAss");
    py::class_<Map>(m, "Map")
        .def(py::init<>())
        .def("add_box", &Map::add_box)
        .def("add_circle", &Map::add_circle)
        .def("setup", &Map::setup)
        .def("get", &Map::get);
        
    // m.def("tmp_set_map", &tmp_set_map, "PAss");
}
