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

struct Map{
    virtual double get_meas(double, double, double){
        py::print(__func__, "is not implemented");
        return 0.0;
    }
    virtual double get_meas_prob(double, double, double, double){
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


    virtual double get_sizeX(){
        py::print(__func__, "is not implemented");
        return 0;
    }
    virtual double get_sizeY(){
        py::print(__func__, "is not implemented");
        return 0;
    }
    // virtual double get_meas_interval(double, double, double, double){
    //     py::print(__func__, "is not implemented");
    //     return 0.0;
    // }

    virtual ~Map() = default;
};

struct PrimitiveMap : public Map{
private:
    std::default_random_engine gen1, gen2;

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

    PrimitiveMap(double bound):
            gen1(std::random_device()()),
            gen2(std::random_device()()){
        width = bound;
        height = bound;
        add_line(-1.,0.,bound);
        add_line(0.,-1.,bound);
        add_line(1.,0.,0.);
        add_line(0.,1.,0.);
    }

    static PrimitiveMap* create(double bound){
        // return std::make_shared<FastMap>(bound);
        return new PrimitiveMap(bound);
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
        double meas = 1000;
        for (auto& o:objs){
            auto m = o->get_dist(x,y,ori);
            if (m>0 && m<meas) meas = m;
        }
        return meas;
    }

    double get_meas_prob(double real, double x, double y, double ori) override{
        double meas = get_meas(x,y,ori);
        // double sig = (double)width*sqrt(2)/3;
        double sig = (double)width*sqrt(2)/30;
        // double sig = (double)width*sqrt(2)/18;
        double p = (meas-real)/sig;
        return exp(-p*p/2)/sig/sqrt(PI2);
    }

    bool is_valid(double x, double y) override{
        bool ret = true;
        for (auto& o:objs){
            ret = ret && o->is_valid(x,y);
            if(!ret) break;
        }
        return ret;
    }

    void set_random_pos(double &x, double &y) override{
        std::uniform_real_distribution<double> w(0.0,(double)width),
                                                h(0.0,(double)height);
        do{
            x = w(gen1);
            y = h(gen2);
        }while(!is_valid(x,y));
    }


    virtual double get_sizeX() override{
        return width;
    }
    virtual double get_sizeY() override{
        return height;
    }
};



struct HeightMap : public Map{
private:
    std::default_random_engine gen1, gen2;
    std::vector<std::vector<double>> grid;
    double gmin, gmax;

public:
    HeightMap(const py::array_t<double>& grid_):
            gen1(std::random_device()()),
            gen2(std::random_device()()){
        auto r = grid_.unchecked<2>();
        grid.resize(r.shape(0));
        for (auto& g:grid)
            g.resize(r.shape(1));

        gmin = 10000;
        gmax = -gmin;
        for (py::ssize_t x = 0; x < r.shape(0); ++x)
            for (py::ssize_t y = 0; y < r.shape(1); ++y){
                grid[x][y] = r(x,y);
                if (grid[x][y]>gmax) gmax = grid[x][y];
                if (grid[x][y]<gmin) gmin = grid[x][y];
            }
    }

    static HeightMap* create(const py::array_t<double>& grid_){
        // return std::make_shared<HeightMap>(grid_);
        return new HeightMap(grid_);
    }

    double get_meas(double x, double y, double ori) override{
        if (!is_valid(x, y)) return -10000.0;
        return grid[int(x)][int(y)];
    }

    void set_random_pos(double& x, double& y) override{
        std::uniform_real_distribution<double> w(0.0,(double)grid.size()-1.), // TODO 100 bo hack
                                                h(0.0,(double)grid[0].size()-1.);
        do{
            x = w(gen1);
            y = h(gen2);
        }while(!is_valid(x,y));
    
    }

    bool is_valid(double x, double y) override{
        if (x<0.0 || y< 0.0 || x >= grid.size() || y >= grid[0].size())
            return false;
        return true;
    }

    std::vector<std::vector<double>> get_grid(){
        return grid;
    }

    double get_meas_prob(double real, double x, double y, double ori) override{
        double meas = get_meas(x,y,ori);
        double sig = (gmax-gmin)/3;
        double p = (meas-real)/sig;
        return exp(-p*p/2)/sig/sqrt(PI2);
    }

    double get_sizeX() override{
        return grid.size();
    }

    double get_sizeY() override{
        return grid[0].size();
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

py::array_t<robot_2d> get_random_pop(
    std::shared_ptr<Map> map,size_t N){
    auto ret = py::array_t<robot_2d>(N);
    py::buffer_info buf = ret.request();
    auto ptr = static_cast<robot_2d*>(buf.ptr);
    
    std::default_random_engine gen;
    std::uniform_real_distribution<double>
        ro(0.0,PI2), vo(MAGIC_VELOCITY_MIN, MAGIC_VELOCITY_MAX);
    for(size_t i=0; i < N; ++i){
        ptr[i] = {0.0, 0.0, ro(gen), vo(gen)};
        map->set_random_pos(ptr[i].x, ptr[i].y);
    }
    return ret;
}

py::array_t<robot_2d> get_linear_pop(size_t N){
    auto ret = py::array_t<robot_2d>(N);
    py::buffer_info buf = ret.request();
    auto ptr = static_cast<robot_2d*>(buf.ptr);
    
    for(size_t i=1; i < N-1; ++i){
        ptr[i] = robot_2d((double)i, (double)N/2, 0.0, 0.0);
    }
    return ret;
}

void drift_state(std::shared_ptr<Map> map, 
    robot_2d& state, double dori, double dvel){
    state.vel += dvel;
    state.ori += dori;

    double tmpx = state.x + cos(state.ori)*state.vel;
    double tmpy = state.y + sin(state.ori)*state.vel;

    if (!map->is_valid(tmpx,tmpy))
        state.ori += PI;
    // if (map.get_meas(state.x,state.y,state.ori)<state.vel)
    //     state.ori += PI;
    state.x += cos(state.ori)*state.vel;
    state.y += sin(state.ori)*state.vel;
}

void drift_pop(std::shared_ptr<Map> map,
    py::array_t<robot_2d> a_pop, double dori, double dvel,
    double var_ori, double var_vel){
    auto pop = a_pop.mutable_unchecked<1>();

    std::random_device bese_gen;
    std::default_random_engine gen1(bese_gen());
    std::default_random_engine gen2(bese_gen());
    std::uniform_real_distribution<double> rand_ori(-var_ori,var_ori), 
                                            rand_vel(-var_vel,var_vel);

    for(size_t i=0; i < pop.shape(0); ++i){
        pop(i).ori += rand_ori(gen1);
        pop(i).vel += rand_vel(gen2);
        drift_state(map, pop(i), dori, dvel);
    }
}

py::array_t<double> update_weights(
    std::shared_ptr<Map> map,
    double real, py::array_t<robot_2d> a_pop,
    py::array_t<double> a_weights){

    auto pop = a_pop.mutable_unchecked<1>();
    auto a_ret = py::array_t<double>(pop.shape(0));
    auto ret = a_ret.mutable_unchecked<1>();
    auto weights = a_weights.mutable_unchecked<1>();
    
    double sum = 0.0;
    for(size_t i=0; i < pop.shape(0); ++i){
        // py::print(i);
        ret(i) = weights(i)*map->get_meas_prob(real, pop(i).x, pop(i).y, pop(i).ori);
        if(isnan(ret(i)) || isnan(weights(i))){
            py::print(__func__,i,ret(i),weights(i),map->get_meas_prob(real, pop(i).x, pop(i).y, pop(i).ori));
        }
        sum += ret(i);
    }
    // py::print(__func__,"sum",sum,"flush"_a = true);
    for(size_t i=0; i < pop.shape(0); ++i){
        ret(i) /= sum;
    }
    return a_ret;
}

py::array_t<double> get_uniform_weights(size_t N){
    auto a_ret = py::array_t<double>(N);
    auto ret = a_ret.mutable_unchecked<1>();
    
    const double w = 1.0/N;
    for(size_t i=0; i < ret.shape(0); ++i){
        ret(i) = w;
    }
    return a_ret;
}

robot_2d get_est(py::array_t<robot_2d> a_pop,
    py::array_t<double> a_weights){
    auto pop = a_pop.mutable_unchecked<1>();
    auto weights = a_weights.mutable_unchecked<1>();

    double x=0,y=0,orix=0,oriy=0,vel=0;
    for(size_t i = 0; i<pop.shape(0);++i){
        orix+=cos(pop(i).ori)*weights(i);
        oriy+=sin(pop(i).ori)*weights(i);
        x+=pop(i).x*weights(i);
        y+=pop(i).y*weights(i);
        vel+=pop(i).vel*weights(i);
    }
    double sum = 0;
    for(size_t i = 0; i<weights.shape(0);++i) sum+=weights(i);
    x /= sum;
    y /= sum;
    orix /= sum;
    oriy /= sum;
    vel /= sum;
    return robot_2d(x,y,atan2(oriy,orix),vel);
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

// #include <list>
#include <functional>

double get_dist(robot_2d& r1, robot_2d& r2){
    // return std::sqrt(
    //     std::pow(r1.x-r2.x,2)+
    //     std::pow(r1.y-r2.y,2));
    return std::sqrt(
        std::pow(r1.x-r2.x,2)+
        std::pow(r1.y-r2.y,2)+
        std::pow(r1.ori-r2.ori,2)+
        std::pow(r1.vel-r2.vel,2));
}

size_t get_new_N(
// std::vector<double> get_new_N(
    std::shared_ptr<Map> map,
    py::array_t<robot_2d> a_pop,
    py::array_t<double> a_weights,
    double meas, double alpha, double gamma = 0.1){
    auto pop = a_pop.mutable_unchecked<1>();
    auto weights = a_weights.mutable_unchecked<1>();

    // list<int> S;
    std::vector<int> S;
    for (int i = 0; i < pop.shape(0); ++i) S.push_back(i);
    // double gamma = 1.0;

    auto get_zeta = [&](){
        double x=0,y=0,orix=0,oriy=0,vel=0;
        for(size_t i = 0; i<S.size();++i){
            orix+=cos(pop(S[i]).ori)*weights(S[i]);
            oriy+=sin(pop(S[i]).ori)*weights(S[i]);
            x+=pop(S[i]).x*weights(S[i]);
            y+=pop(S[i]).y*weights(S[i]);
            vel+=pop(S[i]).vel*weights(S[i]);
        }
        double sum = 0;
        for(size_t i = 0; i<S.size();++i) sum+=weights(S[i]);
        x /= sum;
        y /= sum;
        orix /= sum;
        oriy /= sum;
        vel /= sum;
        return std::abs(map->get_meas(x,y,atan2(oriy,orix))-meas);
    };

    std::vector<std::pair<
        std::function<bool(int,int)>,
        double>> preds = {
        {[&pop](int a, int b){return pop(a).x < pop(b).x;}, gamma},
        {[&pop](int a, int b){return pop(a).y < pop(b).y;}, gamma},
        {[&pop](int a, int b){return pop(a).ori < pop(b).ori;}, 0.1},
        {[&pop](int a, int b){return pop(a).vel < pop(b).vel;}, 0.1},
    };

    std::vector<double> zeta;
    // double alpha = 0.2;

    for(auto pred:preds){
        std::sort(S.begin(),S.end(),pred.first);
        for (int i = 0; i < S.size() - 1; ++i)
            if(get_dist(pop(S[i]),pop(S[i+1]))<pred.second){ // gamma -> second
                if (weights(i)>weights(i+1)) S.erase(S.begin() + i + 1);
                else S.erase(S.begin() + i);
                i--;
                zeta.push_back(get_zeta());
            }
        // py::print("size reduced to",S.size());
        // py::print(zeta);
    }

    // size_t Nmax = 10000, Nmin = 100, newN = 0;
    size_t Nmax = 1000, Nmin = 100, newN = 0;
    // py::print(std::min_element(zeta.begin(), zeta.end()),
    //     std::max_element(zeta.begin(), zeta.end()));
    if (std::all_of(zeta.begin(), zeta.end(), [&alpha](double e){return e>alpha;}))
        newN = std::uniform_int_distribution<int>(pop.shape(0),Nmax)(
            std::default_random_engine());
    else{
        for(int i=0;i<zeta.size();++i) if(zeta[i]<alpha) newN = i;
        newN = pop.shape(0)-newN;
    }
    if(newN < Nmin) newN = Nmin;
    return newN;

    // return zeta;
}

py::array roulette_wheel_resample(py::array a_pop, py::array_t<double> a_weights,
                                    size_t new_pop_size = 0){
    py::buffer_info buf = a_pop.request();

    // py::array a_new_pop = a_pop;
    // py::buffer_info new_buf = a_new_pop.request();

    auto weights = a_weights.mutable_unchecked<1>();

    std::default_random_engine gen;
    std::discrete_distribution<int> dist(weights.data(0),weights.data(0)+weights.shape(0));

    if (new_pop_size == 0) new_pop_size = buf.shape[0];

    auto new_shape = buf.shape;
    new_shape[0] = new_pop_size;
    py::array a_new_pop(pybind11::dtype(buf), new_shape, buf.strides, nullptr);
    py::buffer_info new_buf = a_new_pop.request();

    for(int i = 0; i < new_pop_size; ++i){
        std::memcpy(
            (char*)new_buf.ptr+i*buf.strides[0],
            (char*)buf.ptr+dist(gen)*buf.strides[0],
            buf.strides[0]);
    }

    return a_new_pop;
}

py::array sus_resample(py::array a_pop, py::array_t<double> a_weights,
                        size_t new_pop_size = 0){
    py::buffer_info buf = a_pop.request();

    // py::array a_new_pop(pybind11::dtype(buf), buf.shape, buf.strides, nullptr);
    // py::buffer_info new_buf = a_new_pop.request();

    auto weights = a_weights.mutable_unchecked<1>();

    std::default_random_engine gen;
    
    if (new_pop_size == 0) new_pop_size = buf.shape[0];

    double sum = 0, wsum=weights(0);
    for (size_t i = 0; i < weights.shape(0); ++i) sum+=weights(i);
    double step = sum/new_pop_size;
    double init = std::uniform_real_distribution<double>(0.,step)(gen);
    size_t j = 0;

    auto new_shape = buf.shape;
    new_shape[0] = new_pop_size;
    py::array a_new_pop(pybind11::dtype(buf), new_shape, buf.strides, nullptr);
    py::buffer_info new_buf = a_new_pop.request();
    // py::print(__func__,"flush"_a = true);
    for (size_t i=0; i < new_pop_size; ++i){
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

void regularize(py::array_t<robot_2d> a_pop, double mutation_prob,
    double stdx, double stdori, double stdvel){
    auto pop = a_pop.mutable_unchecked<1>();

    std::default_random_engine gen1, gen2, gen3;
    std::normal_distribution<double>
        dx(.0, stdx), dori(.0, stdori), dvel(.0, stdvel);
    std::uniform_real_distribution<double> mut(0.0,1.0);

    for (size_t i=0; i < pop.shape(0); ++i){
        if (mutation_prob > mut(gen3)){
            pop(i).x += dx(gen1);
            pop(i).y += dx(gen2);
            pop(i).ori += dori(gen2);
            pop(i).vel += dvel(gen2);
        }
    }
}

#include <boost/numeric/interval.hpp>

using namespace boost;
using namespace boost::numeric;
typedef boost::numeric::interval<
    double,
    boost::numeric::interval_lib::policies<
        boost::numeric::interval_lib::save_state<
            boost::numeric::interval_lib::rounded_transc_std<double> >,
        boost::numeric::interval_lib::checking_base<double> > >
    intd;

struct robot_2di{
    intd x, y, vel, ori;
    robot_2di(double xmin, double xmax,
            double ymin, double ymax,
            double orimin, double orimax,
            double velmin, double velmax)
        : x(xmin, xmax),
        y(ymin, ymax),
        ori(orimin, orimax),
        vel(velmin, velmax){}

    std::vector<robot_2di> split(size_t N, double level = 0.8){
        // static std::default_random_engine gen(std::random_device()());
        static std::default_random_engine gen;
        static std::uniform_real_distribution<double> divp(0.0, 1.0);
        std::vector<robot_2di> ret;
        for (size_t i=0;i<N;++i){
            ret.push_back(*this);
            if (divp(gen)<level){
                if (width(x) > width(y))
                    ret[i].x =
                        (x-x.lower())/double(N) +
                        x.lower() +
                        width(x)*i/N;
                else
                    ret[i].y = (y-y.lower())/double(N) +
                        y.lower() +
                        width(y)*i/N;
                }
            else{
                ret[i].ori =
                        (ori-ori.lower())/double(N) +
                        ori.lower() +
                        width(ori)*i/N;
            }
        }
        return ret;
    }
};

struct BoxParticleFilter{
    std::shared_ptr<Map> map;
    std::vector<robot_2di> pop;
    std::vector<double> weights;

    int axis;

    BoxParticleFilter(std::shared_ptr<Map> map_): map(map_), axis(0){};

    intd get_meas_interval(robot_2di state){
        double mini = 10000, maxi = 0;
        for (int x = state.x.lower(); x < state.x.upper(); ++x){
            for (int y = state.y.lower(); y < state.y.upper(); ++y){
                auto meas = map->get_meas(x, y, median(state.ori));
                mini = std::min(mini, meas);
                maxi = std::max(maxi, meas);
            }
        }
        // double mini = 1000, maxi = 0;
        // for (int x = state.x.lower(); x < state.x.upper(); ++x){
        //     for (int y = state.y.lower(); y < state.y.upper(); ++y){
        //         // py::print('y',y);
        //         if (!map->is_valid(x,y)) continue;
                
        //         for (double ori = state.ori.lower(); ori < state.ori.upper(); ori+=0.01){
        //             auto meas = map->get_meas(x, y, ori);
        //             mini = std::min(mini, meas);
        //             maxi = std::max(maxi, meas);
        //         }
        //     }
        // }
        return intd(mini, maxi);
    }

    double update_weights(double meas, double dm){
        intd real_meas(meas-3*dm, meas+3*dm);
        double sum = 0.0;
        for (size_t i = 0; i < pop.size(); ++i){
            auto meas = get_meas_interval(pop[i]);
            auto r = intersect(real_meas, meas);
            double A = 0;
            if (width(r) > 1e-10) A = width(r)/width(meas);
            weights[i]*=A;
            sum += weights[i];
        }

        double effN = 0.0;
        for (size_t i = 0; i < pop.size(); ++i){
            weights[i]/=sum;
            effN+=weights[i]*weights[i];
        }
        return 1/effN; 
    }

    void init_pop(size_t sqrtN){
        double sizeX = map->get_sizeX()/sqrtN;
        double sizeY = map->get_sizeY()/sqrtN;

        pop.clear();
        weights.clear();

        for(size_t x = 0; x < sqrtN; ++x){
            for(size_t y = 0; y < sqrtN; ++y){
                pop.emplace_back(sizeX*x,sizeX*(x+1),
                                sizeY*y,sizeY*(y+1),
                                PI/4-0.1, PI/4+0.1,
                                // 0.0, PI2,
                                // 5, 10);
                                10.0-1.0, 10.0+1.0);
                weights.push_back(1.0/sqrtN/sqrtN);
            }
        }
    }

    void reinit_pop(){

        double minx = map->get_sizeX(), maxx = 0;
        double miny = map->get_sizeY(), maxy = 0;
        double minori = 1.0e9, maxori = -1.0e9;
        double minvel = 1.0e9, maxvel = -1.0e9;

        // for(const auto& p:pop){
        for(size_t i = 0; i < pop.size(); ++i){
            const auto& p = pop[i];
            minx = std::min(p.x.lower(),minx);
            maxx = std::max(p.x.upper(),maxx);

            miny = std::min(p.y.lower(),miny);
            maxy = std::max(p.y.upper(),maxy);

            minori = std::min(p.ori.lower(),minori);
            maxori = std::max(p.ori.upper(),maxori);

            minvel = std::min(p.vel.lower(),minvel);
            maxvel = std::max(p.vel.upper(),maxvel);
            weights[i] = 1.0/pop.size();
        }

        py::print(__func__,
            minx,maxx,
            miny,maxy,
            minori,maxori,
            minvel,maxvel);

        auto bb = robot_2di(minx, maxx,
                            miny, maxy,
                            minori, maxori,
                            minvel, maxvel);

        size_t sqrtN = (size_t)std::sqrt(pop.size());
        auto tmp_pop = bb.split(sqrtN, 1.0);

        std::vector<robot_2di> new_pop;
        for(auto p:tmp_pop){
            auto new_ints = p.split(sqrtN, 1.0);
            new_pop.insert(new_pop.end(),
                new_ints.begin(),
                new_ints.end());
        }
        pop = new_pop;
    }

    double get_coeff(){
        intd x, y;
        double sum;
        for (const auto& p:pop){
            x = hull(x, p.x);
            y = hull(y, p.y);
            sum += width(p.x)*width(p.y);
        }
        return sqrt(sum/pop.size()/width(x)/width(y));
    }

    void drift(double dori, double stdori){
        auto u = intd(dori-3*stdori, dori+3*stdori);
        double sizeX = map->get_sizeX();
        double sizeY = map->get_sizeY();

        for (auto& p:pop){
            p.ori = p.ori + u;
            p.x += cos(p.ori)*p.vel;
            p.y += sin(p.ori)*p.vel;
            p.x.set(max(p.x.lower(), double(0.0)),min(p.x.upper(),sizeX));
            p.y.set(max(p.y.lower(), double(0.0)),min(p.y.upper(),sizeY));
        }
    }

    py::array_t<double> get_pop(){
        std::vector<size_t> shape{pop.size(),8};
        auto a_ret = py::array_t<double>(shape);
        auto ret = a_ret.mutable_unchecked<2>();

        for (size_t i = 0; i < pop.size(); ++i){
            ret(i,0) = pop[i].x.lower();
            ret(i,1) = pop[i].x.upper();
            ret(i,2) = pop[i].y.lower();
            ret(i,3) = pop[i].y.upper();
            ret(i,4) = pop[i].ori.lower();
            ret(i,5) = pop[i].ori.upper();
            ret(i,6) = pop[i].vel.lower();
            ret(i,7) = pop[i].vel.upper();
        }
        return a_ret;
    }
    py::array_t<double> get_est(){
        std::vector<size_t> shape{3};
        auto a_ret = py::array_t<double>(shape);
        auto ret = a_ret.mutable_unchecked<1>();

        double x=0,y=0,orix=0,oriy=0,vel=0;
        for(size_t i = 0; i<pop.size();++i){
            double cx = (pop[i].x.upper() + pop[i].x.lower())/2;
            double cy = (pop[i].y.upper() + pop[i].y.lower())/2;
            double cvel = (pop[i].vel.upper() + pop[i].vel.lower())/2;
            // orix+=cos(pop[i].ori)*weights[i];
            // oriy+=sin(pop[i].ori)*weights[i];
            x+=cx*weights[i];
            y+=cy*weights[i];
            vel+=cvel*weights[i];
            // py::print("w",weights[i]);
        }
        double sum = 0;
        for(size_t i = 0; i<weights.size();++i) sum+=weights[i];
        x /= sum;
        y /= sum;
        // orix /= sum;
        // oriy /= sum;
        vel /= sum;

        ret(0) = x;
        ret(1) = y;
        // ret(2) = ori;
        // ret(3) = vel;
        ret(2) = vel;
        // py::print(__func__,x,y,sum);
        return a_ret;
    }

    void resample(){
        std::default_random_engine gen;
        
        std::vector<robot_2di> new_pop;

        double sum = 0, wsum = weights[0];
        // double sum = 0, wsum = 0;
        for (size_t i = 0; i < weights.size(); ++i) sum+=weights[i];
        double step = sum/pop.size();
        double init = std::uniform_real_distribution<double>(0.,step)(gen);
        size_t j = 0;

        size_t counter = 0, ind = 0, cumsum = 0;
        // py::print("stats:",j,init, step, sum);

        std::vector<size_t> inds;
        for (size_t i=0; i < pop.size(); ++i){
            double lw = init+step*i;

            while(wsum<lw){ 
                j++;
                wsum+=weights[j];
            }

            inds.push_back(j);
        }

        for (size_t i=0; i < pop.size(); ++i){
            size_t counter = std::count(
                inds.begin(),
                inds.end(),
                i);
            if (counter==0) continue;
            auto new_ints = pop[i].split(counter);
            new_pop.insert(new_pop.end(),
                new_ints.begin(),
                new_ints.end());
        }

        // py::print("new size:",new_pop.size(),counter,j,weights[j]);
        pop = new_pop;

        axis+=1;
        for (size_t i = 0; i < weights.size(); ++i)
            weights[i] = 1.0/weights.size();
        // py::print("-----done");
    }
};


PYBIND11_MODULE(PFlib, m){
    m.doc() = "particle filter lib";

    m.def("roulette_wheel_resample", roulette_wheel_resample,
        py::arg("pop"), py::arg("weights"), py::arg("weights") = 0);
    m.def("sus_resample", sus_resample,
        py::arg("pop"), py::arg("weights"), py::arg("weights") = 0);

    m.def("as_array", as_array);
    m.def("get_uniform_weights", get_uniform_weights);
    m.def("get_est", get_est);
    m.def("update_weights", update_weights);
    m.def("drift_state", drift_state);
    m.def("drift_pop", drift_pop);
    m.def("get_random_pop", get_random_pop);
    m.def("get_linear_pop", get_linear_pop);
    m.def("get_new_N", get_new_N,
        "Particle filtering with adaptive number of particles, doi=10.1109/AERO.2011.5747439");
    m.def("regularize", regularize,
        "actualy covered in drift");

    py::class_<robot_2d>(m,"robot_2d")
        .def(py::init<>())
        .def(py::init<double,double,double,double>())
        .def_readwrite("x", &robot_2d::x)
        .def_readwrite("y", &robot_2d::y)
        .def_readwrite("ori", &robot_2d::ori)
        .def_readwrite("vel", &robot_2d::vel);
    PYBIND11_NUMPY_DTYPE(robot_2d, x, y, ori, vel);

    py::class_<BoxParticleFilter>(m,"BoxParticleFilter")
        .def(py::init<std::shared_ptr<Map>>())
        .def("init_pop",&BoxParticleFilter::init_pop)
        .def("reinit_pop",&BoxParticleFilter::reinit_pop)
        .def("update_weights",&BoxParticleFilter::update_weights)
        .def("drift",&BoxParticleFilter::drift)
        .def("get_est",&BoxParticleFilter::get_est)
        .def("resample",&BoxParticleFilter::resample)
        .def("get_coeff",&BoxParticleFilter::get_coeff)
        .def("get_pop",&BoxParticleFilter::get_pop);

    py::class_<Map, std::shared_ptr<Map>>(m, "Map");

    py::class_<PrimitiveMap, Map, std::shared_ptr<PrimitiveMap>>(m, "PrimitiveMap")
        .def(py::init(&PrimitiveMap::create))
        .def("get_grid", &PrimitiveMap::get_grid)
        .def("add_line", &PrimitiveMap::add_line)
        .def("add_circle", &PrimitiveMap::add_circle)
        .def("get_meas", py::vectorize(&PrimitiveMap::get_meas));

    py::class_<HeightMap, Map, std::shared_ptr<HeightMap>>(m, "HeightMap")
        .def(py::init(&HeightMap::create))
        .def("get_meas_prob", &HeightMap::get_meas_prob)
        .def("get_grid", &HeightMap::get_grid)
        .def("get_meas", py::vectorize(&HeightMap::get_meas));
}



/*
Jakie badania

*/