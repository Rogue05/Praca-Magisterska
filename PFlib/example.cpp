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

template<typename ModelT, typename StateT>
struct ParticleFilter{
    std::default_random_engine gen;

    ModelT* model;
    
    std::vector<StateT> pop;
    std::vector<double> weights;

    py::array get_pop(){
        // py::print(__func__,"flush"_a=true);
        // vector<vector<double>> ret;
        // for (const auto& p : pop) ret.push_back({p.x,p.y,p.ori});
        // return py::cast(ret);
        return model->translate_pop(pop);
    }
    

    py::array get_weights(){
        // py::print(__func__,"flush"_a=true);
        return py::cast(weights);
    }

    void set_model(ModelT &model_){
        model=&model_;
    }

    void update_weights(){
        // py::print(__func__,"flush"_a=true);
        weights.resize(pop.size());

        double sum=0;
        for (size_t i = 0;i<pop.size();++i){
            weights[i] = weights[i]*model->get_state_prob(pop[i]);
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
        pop = model->get_random_states(N);
        py::print("initialized pop",N);
        
        weights.resize(pop.size());
        for (auto & w : weights) w = 1./N;
        py::print("initialized weights");
    }

    py::tuple get_est(){
        // py::print(__func__,"flush"_a=true);
        return model->get_est(pop, weights);
        // auto est = model->get_est(pop, weights);
        // return py::make_tuple(est.x,est.y,est.ori,est.vel);
    }
    
    void resample(RESAMPLE_TYPE type){
        // py::print(__func__,"flush"_a=true);
        std::vector<StateT> new_pop;
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

    void drift(){
        // py::print(__func__,"flush"_a=true
        model->drift_states(pop);
    }
};


PYBIND11_MODULE(PFlib, m){
    m.doc() = "pybind11 example plugin"; // optional module docstring

    py::enum_<RESAMPLE_TYPE>(m, "RESAMPLE_TYPE")
        .value("ROULETTE_WHEEL",RESAMPLE_TYPE::ROULETTE_WHEEL)
        .value("SUS",RESAMPLE_TYPE::SUS).export_values();

    py::class_<ParticleFilter<Model,Model::State>>(m,"ParticleFilter")
        .def(py::init<>())
        .def("get_effective_N", &ParticleFilter<Model,Model::State>::get_effective_N)
        .def("get_est",&ParticleFilter<Model,Model::State>::get_est)
        .def("setup", &ParticleFilter<Model,Model::State>::setup)
        .def("get_pop", &ParticleFilter<Model,Model::State>::get_pop)
        .def("get_weights", &ParticleFilter<Model,Model::State>::get_weights)
        .def("set_model", &ParticleFilter<Model,Model::State>::set_model)
        .def("update_weights", &ParticleFilter<Model,Model::State>::update_weights)
        .def("resample", &ParticleFilter<Model,Model::State>::resample)
        .def("drift", &ParticleFilter<Model,Model::State>::drift);

    py::class_<ParticleFilter<PlaneModel,PlaneModel::State>>(m,"PlaneParticleFilter")
        .def(py::init<>())
        .def("get_effective_N", &ParticleFilter<PlaneModel,PlaneModel::State>::get_effective_N)
        .def("get_est",&ParticleFilter<PlaneModel,PlaneModel::State>::get_est)
        .def("setup", &ParticleFilter<PlaneModel,PlaneModel::State>::setup)
        .def("get_pop", &ParticleFilter<PlaneModel,PlaneModel::State>::get_pop)
        .def("get_weights", &ParticleFilter<PlaneModel,PlaneModel::State>::get_weights)
        .def("set_model", &ParticleFilter<PlaneModel,PlaneModel::State>::set_model)
        .def("update_weights", &ParticleFilter<PlaneModel,PlaneModel::State>::update_weights)
        .def("resample", &ParticleFilter<PlaneModel,PlaneModel::State>::resample)
        .def("drift", &ParticleFilter<PlaneModel,PlaneModel::State>::drift);

    py::class_<Map, std::shared_ptr<Map>>(m, "Map");

    py::class_<FastMap, Map, std::shared_ptr<FastMap>>(m, "FastMap")
        .def(py::init(&FastMap::create))
        .def("get_grid", &FastMap::get_grid)
        .def("add_line", &FastMap::add_line)
        .def("add_circle", &FastMap::add_circle);

    py::class_<HeightMap, Map, std::shared_ptr<HeightMap>>(m, "HeightMap")
        .def(py::init(&HeightMap::create))
        .def("get_meas_prob", &HeightMap::get_meas_prob)
        .def("get_grid", &HeightMap::get_grid);

    py::class_<Model>(m, "Model")
        .def(py::init<double, double, double, double, double, double, double>())
        .def("get_real", &Model::get_real)
        .def("update_meas", &Model::update_meas)
        .def("set_map", &Model::set_map)
        .def("update", &Model::update);

    py::class_<PlaneModel>(m, "PlaneModel")
        .def(py::init<double, double, double, double, double, double, double>())
        .def("get_real", &PlaneModel::get_real)
        .def("update_meas", &PlaneModel::update_meas)
        .def("set_map", &PlaneModel::set_map)
        .def("update", &PlaneModel::update);

    // py::class_<Model::Measurment>(m, "Model.Measurment")
    //     .def("randomize",&Model::Measurment::randomize)
    //     .def("get",&Model::Measurment::get);
}
