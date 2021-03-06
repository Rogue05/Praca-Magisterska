#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <random>
#include <vector>
// #include <cmath>

#include <iostream>

#include "Map.hpp"
#include "Model.hpp"

using namespace std;
namespace py = pybind11;
using namespace pybind11::literals;


enum RESAMPLE_TYPE{
    ROULETTE_WHEEL,
    SUS
};

struct ParticleFilter{
    std::default_random_engine gen;

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
            pop.push_back(model->get_random_state());
        }
        py::print("initialized pop",N);
        
        weights.resize(pop.size());
        for (auto & w : weights) w = 1./N;
        py::print("initialized weights");
    }

    py::tuple get_est(){
        // py::print(__func__,"flush"_a=true);
        auto est = model->get_est(pop, weights);
        return py::make_tuple(est.x,est.y,est.ori,est.vel);
    }

    double get_est_meas(){
        // py::print(__func__,"flush"_a=true);
        return model->_get_meas(model->get_est(pop, weights)).dist
                - model->get_meas().dist;
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

    // void drift(double dori, double sigv, double sigori){
    void drift(double dori){
        // py::print(__func__,"flush"_a=true);
        for(auto &p : pop){
            model->drift(p, dori);
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
        .def("get_est_meas", &ParticleFilter::get_est_meas)
        .def("drift", &ParticleFilter::drift);

    py::class_<Map, std::shared_ptr<Map>>(m, "Map");

    py::class_<FastMap, Map, std::shared_ptr<FastMap>>(m, "FastMap")
        .def(py::init(&FastMap::create))
        .def("get_grid", &FastMap::get_grid)
        .def("add_line", &FastMap::add_line)
        .def("add_circle", &FastMap::add_circle);

    py::class_<HeightMap, Map, std::shared_ptr<HeightMap>>(m, "HeightMap")
        .def(py::init(&HeightMap::create));

    py::class_<Model>(m, "Model")
        .def(py::init<double, double, double, double, double, double, double>())
        .def("set", &Model::set)
        .def("get", &Model::get)
        .def("get_meas", &Model::get_meas)
        .def("set_map", &Model::set_map)
        .def("update", &Model::update);

    py::class_<Model::Measurment>(m, "Model.Measurment")
        .def("randomize",&Model::Measurment::randomize);
        // .def("get",&Model::Measurment::get);
}
