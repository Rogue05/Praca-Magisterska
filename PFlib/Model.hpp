#pragma once

#include <cmath>
#include <random>
#include <memory>

#include <pybind11/pybind11.h> // print

struct Model{
    Map* map;
    double maxvel;
    std::default_random_engine gen;
    std::normal_distribution<double> dvn, dorin;
    std::uniform_real_distribution<double> ro, rv;
    double dv, dori;
    double meas;

    struct State{
        double x,y,ori,vel;
        State(double x_, double y_, double ori_, double vel_)
            :x(x_), y(y_), ori(ori_), vel(vel_){}
        State(): State(0.,0.,0.,0.){};

        void update(Map* map){
            if (map->get_meas(x,y,ori)<vel) ori += PI;
            x += cos(ori)*vel;
            y += sin(ori)*vel;
        }
    } real_state;

    Model(): real_state(0,0,0,0){}
    Model(double x_, double y_, double z_,
        double vel_, double maxvel_, double sigv,
        double sigori):
        real_state(x_, y_, z_, vel_),
        dvn(0,sigv),
        dorin(0,sigori),
        ro(0.0,PI2),
        rv(0.0,maxvel_),
        meas(0){}

    py::array translate_pop(std::vector<State>& pop){
        // py::print(__func__,"flush"_a=true);
        std::vector<std::vector<double>> ret;
        for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori});
        return py::cast(ret);
    }

    std::vector<State> get_random_states(size_t N){
        std::vector<State> pop;
        pop.resize(0);
        for (size_t i=0;i<N;++i){
            pop.emplace_back(0, 0, ro(gen), rv(gen));
            map->set_random_pos(pop[i].x,pop[i].y);
        }
        return pop;
    }

    void update_meas(double dm){
        meas = map->get_meas(real_state.x,real_state.y,real_state.ori)+dm;
    }

    double get_state_prob(const State &st){
        if (!is_valid(st)) return false;
        return map->get_meas_prob(meas, st.x, st.y, st.ori);
    }

    void set_map(Map* mapc){
        map = mapc;
    }

    py::array get_real(){
        // py::print(__func__,"flush"_a=true);
        std::vector<double> ret{
            real_state.x,
            real_state.y,
            real_state.ori,
            real_state.vel};
        return py::cast(ret);
    }

    py::tuple get_est(std::vector<State>& pop, std::vector<double>& weights){
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
        return py::make_tuple(x,y,atan2(oriy,orix),vel);
        // return {x,y,atan2(oriy,orix),vel};
    }

    void update(double dori_, double dvel_){
        dv = dvel_;
        dori = dori_;

        real_state.vel += dvel_;
        real_state.ori += dori_;
        real_state.update(map);
    }

    void drift_states(std::vector<State>& pop){
        for(auto &p : pop){
            p.vel += dv + dvn(gen);
            p.ori += dori + dorin(gen);
            p.update(map);
        }
    }

    bool is_valid(const State& st){
        return map->is_valid(st.x,st.y);
    }
};


struct PlaneModel{
    Map* map;
    double maxvel;
    std::default_random_engine gen;
    std::normal_distribution<double> dvn, dorin;
    std::uniform_real_distribution<double> ro, rv;
    double dv, dori;
    double meash, measori;

    struct State{
        double x,y,ori,vel;
        State(double x_, double y_, double ori_, double vel_)
            :x(x_), y(y_), ori(ori_), vel(vel_){}
        State(): State(0.,0.,0.,0.){};

        void update(){
            x += cos(ori)*vel;
            y += sin(ori)*vel;
        }
    } real_state;

    PlaneModel(): real_state({0,0,0,0}){}
    PlaneModel(double x_, double y_, double z_,
        double vel_, double maxvel_, double sigv,
        double sigori):
        real_state(x_, y_, z_, vel_),
        dvn(0,sigv),
        dorin(0,sigori),
        ro(0.0,PI2),
        rv(0.0,maxvel_),
        meash(0),measori(0){}

    py::array translate_pop(std::vector<State>& pop){
        // py::print(__func__,"flush"_a=true);
        std::vector<std::vector<double>> ret;
        for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori,p.vel});
        return py::cast(ret);
    }

    std::vector<State> get_random_states(size_t N){
        std::vector<State> pop;
        pop.resize(0);
        for (size_t i=0;i<N;++i){
            pop.emplace_back(0, 0, real_state.ori, rv(gen));
            map->set_random_pos(pop[i].x,pop[i].y);
        }
        return pop;
    }

    void update_meas(double dh, double dori){
        meash = map->get_meas(real_state.x,real_state.y,real_state.ori)+dh;
        measori = real_state.ori + dori;
    }

    double get_state_prob(const State &st){
        if (!is_valid(st)) return false;
        // return map->get_meas_prob(meash, st.x, st.y, st.ori);
        double hprob = map->get_meas_prob(meash, st.x, st.y, st.ori);
        
        // double p = std::min()
        double b = std::max(st.ori,measori), a = std::min(st.ori, measori);
        double p = std::min(b-a,a+PI2-b)/PI;
        double oriprob = 1. - p;
        // py::print(__func__,real_state.ori,measori,oriprob);
        return hprob;
    }

    void set_map(Map* mapc){
        map = mapc;
    }

    py::array get_real(){
        // py::print(__func__,"flush"_a=true);
        std::vector<double> ret{
            real_state.x,
            real_state.y,
            real_state.ori,
            real_state.vel};
        return py::cast(ret);
    }

    py::tuple get_est(std::vector<State>& pop, std::vector<double>& weights){
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
        return py::make_tuple(x,y,atan2(oriy,orix),vel);
        // return {x,y,atan2(oriy,orix),vel};
    }

    void update(double dori_, double dvel_){
        dv = dvel_;
        dori = dori_;

        real_state.vel += dvel_;
        real_state.ori += dori_;
        real_state.update();
    }

    void drift_states(std::vector<State>& pop){
        for(auto &p : pop){
            p.vel += dv + dvn(gen);
            p.ori += dori + dorin(gen);
            p.update();
        }
    }

    bool is_valid(const State& st){
        return map->is_valid(st.x,st.y);
    }
};
