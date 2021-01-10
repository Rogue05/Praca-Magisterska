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

vector<vector<bool>> grid;

struct State{
    double x,y,ori;
};

struct Model : State{
    double vel;
} model;

std::vector<State> pop;
std::vector<double> weights;

std::default_random_engine gen;

const double PI  = 3.141592653589793238463;
const double PI2  = 2*PI;

double _get_meas(double x, double y, double ori){
    int i = 0;
    double dx = 1,ret=0;

    while(dx>1e-3){
        while(grid
            [x+(cos(ori)*(dx+ret))]
            [y+(sin(ori)*(dx+ret))]==false) {
                // ++i;
                ret+=dx;
            }
        dx/=2;
    }
    // i--;


    return ret;
    // return dx*i;
}

void init_pop(size_t N){
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

// vector<vector<double>> get_pop(){
py::array get_pop(){
    // py::print(__func__,"flush"_a=true);
    vector<vector<double>> ret;
    for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori});
    // return ret;
    // py::array tmp = py::cast(ret);
    // return tmp;
    return py::cast(ret);
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

        weights[i] = weights[i]*(1.-(double)fabs(meas-m)/grid.size());
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

// void set_weights(double meas){
//     // py::print("Resizing");
//     weights.resize(pop.size());
//     // py::print("Resized");
//     for (size_t i = 0;i<pop.size();++i){
//     // py::print("i:",i);
//         if (pop[i].x < 0 || pop[i].x > grid.size() ||
//             pop[i].y < 0 || pop[i].y > grid[0].size() ||
//             grid[(int)pop[i].x][(int)pop[i].y]){
//                 weights[i]=0;
//                 continue;
//             }
//         double m = _get_meas(pop[i].x,pop[i].y,pop[i].ori);
//     // py::print("m:",m);
//         weights[i] = grid.size()-fabs(meas-m);
//         // weights[i] = fabs(meas-m);
//     // py::print("weights[i]:",weights[i]);
//     }
// }

enum RESAMPLE_TYPE{
    ROULETTE_WHEEL,
    SUS
};

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
            // py::print('ADD',j);
        }
        // py::print("SUS:",pop.size(),new_pop.size());
    }
    pop = new_pop;
    for (auto& w: weights) w=1./pop.size();
}

void drift(double dori){
    // py::print(__func__,"flush"_a=true);
    // py::print("Drifting hehe");
    for(auto &p : pop){
        // py::print("elem:",p.x,p.y,"flush"_a=true);
        // py::print("poss:",(int)p.x,(int)p.y,(int)grid[(int)p.x][(int)p.y],"flush"_a=true);
        // // py::print("poss:",(int)(p.x < 0),(int)p.y,"flush"_a=true);
        if (p.x < 0 || p.x >= grid.size() ||
            p.y < 0 || p.y >= grid[0].size() ||
            grid[(int)p.x][(int)p.y]) continue;
        // py::print("process","flush"_a=true);
        // py::print("process2","flush"_a=true);
        p.ori += dori;
        double tmpx = p.x + cos(p.ori)*model.vel;
        double tmpy = p.y + sin(p.ori)*model.vel;
        // py::print("adin","flush"_a=true);
        // py::print("poss:",(int)tmpx,(int)tmpy,"flush"_a=true);
        // // py::print("size: ",grid.size());
        // // py::print("size2:",grid[0].size());
        // // py::print("size3:",grid[(int)tmpx].size());
        // py::print("grid:",(int)grid[(int)tmpx][(int)tmpy],"flush"_a=true);
        // // py::print("done");
        
        if (grid[(int)tmpx][(int)tmpy]) p.ori += PI;
        // py::print("dva","flush"_a=true);
        p.x += cos(p.ori)*model.vel;
        p.y += sin(p.ori)*model.vel;
    }
    // py::print(__func__,"done","flush"_a=true);
}

py::tuple get_est(){
    // py::print(__func__,"flush"_a=true);
    double x=0,y=0,orix=0,oriy=0;
    for(size_t i = 0; i<pop.size();++i){
    // for(const auto &p : pop){
        orix+=cos(pop[i].ori)*weights[i];
        oriy+=sin(pop[i].ori)*weights[i];
        x+=pop[i].x*weights[i];
        y+=pop[i].y*weights[i];
    }
    double sum = 0;
    for (const auto& w: weights) sum+=w;
    // py::array_t<double> ret(3);
    // ret[0] = x/pop.size();
    // ret[1] = y/pop.size();
    // ret[2] = atan2(oriy,orix);
    // return ret;
    x /= sum;
    y /= sum;
    orix /= sum;
    oriy /= sum;
    // return py::make_tuple(x/pop.size(),y/pop.size(),atan2(oriy,orix));
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

void add_circle(py::tuple pos,float radius){
    py::print(__func__,"flush"_a=true);
    for(size_t x=0;x<grid.size();++x){
        for(size_t y=0;y<grid.size();++y){
            float x2=(pos[0].cast<float>()-x);
            float y2=(pos[1].cast<float>()-y);
            if(x2*x2+y2*y2 <= radius*radius)
                grid[x][y] = true;
            // else grid[x][y] = false;
            // grid[x][y] = false;
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

py::array get_grid(){
    // py::print(__func__,"flush"_a=true);
// vector<vector<bool>> get_grid(){
    return py::cast(grid);
}

void setup_map(size_t x,size_t y,size_t margin) {
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

PYBIND11_MODULE(PFlib, m){
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::enum_<RESAMPLE_TYPE>(m, "RESAMPLE_TYPE")
        .value("ROULETTE_WHEEL",RESAMPLE_TYPE::ROULETTE_WHEEL)
        .value("SUS",RESAMPLE_TYPE::SUS).export_values();

    m.def("_get_meas", &_get_meas, "PAss");
    m.def("get_effective_N", &get_effective_N, "PAss");
    m.def("get_est",&get_est, "PAss");
    m.def("init_pop", &init_pop, "PAss");
    m.def("get_pop", &get_pop, "PAss");
    m.def("set_model", &set_model, "PAss");
    m.def("update_weights", &update_weights, "PAss");
    m.def("resample", &resample, "PAss");
    m.def("drift", &drift, "PAss");
    m.def("diffuse", &diffuse, "PAss");
    m.def("add_box", &add_box, "PAss");
    m.def("add_circle", &add_circle, "PAss");
    m.def("setup_map", &setup_map, "PAss");
    m.def("get_grid", &get_grid, "PAss");
}
