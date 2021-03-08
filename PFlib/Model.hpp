#pragma once

#include <cmath>
#include <random>
#include <memory>

#include <pybind11/pybind11.h> // print

// struct Model{
//     struct State;
//     struct Measurment;
// };


struct State{
    double x,y,ori,vel;
    void update(Map* map, bool bounded = true){
        if (bounded && map->get_meas(x,y,ori)<vel) ori += PI;
        x += cos(ori)*vel;
        y += sin(ori)*vel;
    }
};

struct Measurment{
    double dist;
    
    void randomize(double var){
        static std::default_random_engine gen;
        static std::normal_distribution<double> dx(1.,var);
        dist *= dx(gen);
    }

    double get(){
        return dist;
    }
};

struct Model{
    Map* map;
    double maxvel;
    State real_state;
    std::default_random_engine gen;
    std::normal_distribution<double> dvn, dorin;
    std::uniform_real_distribution<double> ro, rv;


    Model(): real_state({0,0,0,0}){}
    Model(double x_, double y_, double ori_,
        double vel_, double maxvel_, double sigv, double sigori):
        real_state({x_, y_, ori_, vel_}),
        dvn(0,sigv),
        dorin(0,sigori),
        ro(0.0,PI2),
        rv(0.0,maxvel_){}

    Measurment _get_meas(const State &st){
        Measurment ret;
        ret.dist = map->get_meas(st.x,st.y,st.ori);
        return ret;
    }

    State get_random_state(){
        State st;
        map->set_random_pos(st.x,st.y);
        st.ori = ro(gen);
        st.vel = rv(gen);
        return st;
    }

    Measurment get_meas(){
        return _get_meas(real_state);
    }

    double get_meas_prob(const State &st, const Measurment &meas){
        // // labirynt
        // double m = _get_meas(st).dist;
        // double sig = 1000*sqrt(2)/3;
        // // double sig = 100;
        // double p = (meas.dist-m)/sig;
        // return exp(-p*p/2)/sig/sqrt(PI2);
        return map->get_meas_prob(meas.dist, st.x, st.y, st.ori);
        // return (1.-(double)fabs(meas.dist-m)/1000/sqrt(2.));//TODO hack 1000 bo taka mapa
        
        // // samolot
        // return 1.- (double)fabs(meas.dist -_get_meas(st).dist)/15.;
        // // double m = _get_meas(st).dist;
        // // double sig = 1;
        // // // double sig = 100;
        // // double p = (meas.dist-m)/sig;
        // // return exp(-p*p/2)/sig/sqrt(PI2);
        // // return (1.-(double)fabs(meas.dist-m)/1000/sqrt(2.));//TODO hack 1000 bo taka mapa
    }

    void set_map(Map* mapc){
        map = mapc;
    }

    void set(double x, double y, double ori, double vel_){
        // py::print(__func__,"flush"_a=true);
        real_state.x = x;
        real_state.y = y;
        real_state.ori = ori;
        real_state.vel = vel_;
    }

    py::array get(){
        // py::print(__func__,"flush"_a=true);
        std::vector<double> ret{
            real_state.x,
            real_state.y,
            real_state.ori,
            real_state.vel};
        return py::cast(ret);
    }

    State get_real(){
        return real_state;
    }

    State get_est(std::vector<State>& pop, std::vector<double>& weights){
        double x=0,y=0,orix=0,oriy=0,vel=0;
        for(size_t i = 0; i<pop.size();++i){
            orix+=cos(pop[i].ori)*weights[i];
            oriy+=sin(pop[i].ori)*weights[i];
            x+=pop[i].x*weights[i];
            y+=pop[i].y*weights[i];
            vel+=pop[i].vel*weights[i];
        }
        double sum = 0;
        for (const auto& w: weights) sum+=w;
        x /= sum;
        y /= sum;
        orix /= sum;
        oriy /= sum;
        vel /= sum;
        return {x,y,atan2(oriy,orix),vel};
    }

    void update(double dori, double dvel, bool bounded = true){
        real_state.vel += dvel;
        real_state.ori += dori;
        real_state.update(map, bounded);
    }

    void drift(State &p, double dori, bool bounded = true){
        p.vel += dvn(gen);
        p.ori += dori + dorin(gen);
        p.update(map, bounded);
    }

    bool is_valid(const State& st){
        return map->is_valid(st.x,st.y);
    }
};