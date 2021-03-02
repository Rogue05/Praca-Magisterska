#pragma once

#include <cmath>
#include <random>

#include <pybind11/pybind11.h> // print

// struct Model{
//     struct State;
//     struct Measurment{
//         void randomize();
//     };
//     double get_meas_prob(const State &, const Measurment &);
//     State get_random_state();
//     bool is_valid(const State& st);
//     void drift(State &p, double);
//     void update(double dori, double dvel);
// };

struct Model{
    // double vel;
    FastMap* map;
    std::default_random_engine gen;
    std::normal_distribution<double> dvn, dorin;


    struct State{
        double x,y,ori,vel;
        void update(FastMap* map){
            if (map->get_meas(x,y,ori)<vel) ori += PI;
            x += cos(ori)*vel;
            y += sin(ori)*vel;
        }
    } real_state;

    struct Measurment{
        double dist;
        
        std::default_random_engine gen;
        void randomize(double var){
            // std::uniform_real_distribution<double> dx(0.,var);
            static std::normal_distribution<double> dx(1.,var);
            dist *= dx(gen);
        }
    };

    Model(): real_state({0,0,0,0}){}
    Model(double x_, double y_, double z_, double vel_, double sigv, double sigori):
        real_state({x_, y_, z_, vel_}),
        // vel(vel_),
        dvn(0,sigv),
        dorin(0,sigori)
        {}

    Measurment _get_meas(const State &st){
        Measurment ret;
        ret.dist = map->get_meas(st.x,st.y,st.ori);
        return ret;
    }

    State get_random_state(){
        static std::uniform_real_distribution<double> ro(0.0,PI2),
                                                    rv(0.0,20.0);
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

    void update(double dori, double dvel){
        // drift(real_state,dori);
        real_state.vel += dvel;
        real_state.ori += dori;
        real_state.update(map);
    }

    // void drift(State &p, double dori, double sigv, double sigori){
    void drift(State &p, double dori){
        // static std::normal_distribution<double> dvn(0,sigv), dorin(0,sigori);
        p.vel += dvn(gen);
        p.ori += dori + dorin(gen);
        p.update(map);
        // if (map->get_meas(p.x,p.y,p.ori)<tvel) p.ori += PI;
        // p.x += cos(p.ori)*p.vel;
        // p.y += sin(p.ori)*p.vel;
    }

    bool is_valid(const State& st){
        return map->is_valid(st.x,st.y);
    }
};