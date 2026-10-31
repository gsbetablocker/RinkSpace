#include "rinkspace/local_jet.hpp"
#include "rinkspace/entities.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <utility>
using namespace rinkspace;
static void require(bool ok,const char* message){if(!ok){std::fprintf(stderr,"FAIL %s\n",message);std::exit(1);}}
static void close(double a,double b,double atol,const char* message){require(std::isfinite(a)&&std::isfinite(b)&&std::abs(a-b)<=atol*(1+std::abs(a)+std::abs(b)),message);}
int main(){
    double board[]={.5,.25,.1,.12,1.85,.02};
    std::array<double,ice_tolerance_layout_size> q{};
    const double nominal[]={-.3,.2,0.,.1,.03,.1,.35,.65,.31,.07,-.42,-.43,-.41,-.44,.28,.28,.28,.28,
        0.,.12,.06,-.24,.24,0.,.45,.3,0.,.11,0.,.035,.025,0.,0.,.05,.07,-.23,0.,0.,0.};
    std::copy(nominal,nominal+39,q.begin());
    std::vector<int> codes;for(int i=100;i<132;++i)codes.push_back(i);for(int i=200;i<204;++i)codes.push_back(i);
    for(int i=300;i<312;++i)codes.push_back(i);for(int i=320;i<324;++i)codes.push_back(i);
    for(int i=400;i<412;++i)codes.push_back(i);for(int i=420;i<424;++i)codes.push_back(i);
    int count=0;IceToleranceGeometry<double> g(board,q.data());
    ice_tolerance_margins(g,[&](int id,double margin){require(id==count++,"stable margin indices");require(margin>=-1e-14,"valid seed margins");});
    require(count==ice_tolerance_margin_count,"complete physical face capacity");
    for(int code:codes){double old[8],now[8];int ko,kn;const int n=(code==410||code==411)?8:6;
        require(shared_entity_definition(board,6,q.data(),39,code,ko,old,n,63)==Status::ok,"strict seed");
        require(shared_entity_definition(board,6,q.data(),q.size(),code,kn,now,n,127)==Status::ok,"zero seed");
        require(ko==kn,"primitive family preserved");for(int i=0;i<n;++i)close(now[i],old[i],1e-13,"zero reproduces strict geometry");}
    // Omitted layout columns must be identically inactive away from the
    // nominal zero-deviation seed as well as at it, for every primitive.
    for(int sample=0;sample<4;++sample)for(int code:codes){
        std::array<bool,ice_tolerance_layout_size> support{};
        require(ice_tolerance_layout_support(code,[&](int index){require(index>=0&&index<ice_tolerance_layout_size,"support index range");support[index]=true;}),"all ice supports registered");
        for(int column=0;column<ice_tolerance_layout_size;++column){if(support[column])continue;
            LocalJet b[6],values[ice_tolerance_layout_size],out[8];for(int i=0;i<6;++i)b[i]=board[i];
            for(int i=0;i<ice_tolerance_layout_size;++i){double value=q[i];
                if(sample){if(i>=39)value=.001*std::sin(double(i*7+sample));else value*=1.+.002*std::sin(double(i*3+sample));}
                values[i]={value,i==column?1.:0.};}
            int kind,n=code==410||code==411?8:6;
            require(shared_entity_definition(b,6,values,ice_tolerance_layout_size,code,kind,out,n,127,false)==Status::ok,"structural support geometry");
            for(int i=0;i<n;++i)require(out[i].v==0.,"omitted geometry derivative must be exactly zero");
        }
    }
    const double epsilon=2e-7;
    for(int column=0;column<ice_deviation_count;++column){
        LocalJet b[6],values[ice_tolerance_layout_size];for(int i=0;i<6;++i)b[i]=board[i];
        for(int i=0;i<ice_tolerance_layout_size;++i)values[i]={q[i],i==39+column?1.:0.};
        auto minus=q,plus=q;minus[39+column]-=epsilon;plus[39+column]+=epsilon;
        for(int code:codes){double lo[8],hi[8];LocalJet analytic[8];int k;const int n=(code==410||code==411)?8:6;
            require(shared_entity_definition(b,6,values,q.size(),code,k,analytic,n,127,false)==Status::ok,"directional evaluation");
            require(shared_entity_definition(board,6,minus.data(),minus.size(),code,k,lo,n,127)==Status::ok,"negative FD seed");
            require(shared_entity_definition(board,6,plus.data(),plus.size(),code,k,hi,n,127)==Status::ok,"positive FD seed");
            for(int i=0;i<n;++i)close(analytic[i].v,(hi[i]-lo[i])/(2*epsilon),3e-7,"native deviation derivative");
        }
        std::array<LocalJet,ice_tolerance_margin_count> analytic{};
        IceToleranceGeometry<LocalJet> jg(b,values);ice_tolerance_margins(jg,[&](int i,LocalJet m){analytic[i]=m;});
        double lo[ice_tolerance_margin_count],hi[ice_tolerance_margin_count];
        ice_tolerance_margins(IceToleranceGeometry<double>(board,minus.data()),[&](int i,double m){lo[i]=m;});
        ice_tolerance_margins(IceToleranceGeometry<double>(board,plus.data()),[&](int i,double m){hi[i]=m;});
        for(int i=0;i<ice_tolerance_margin_count;++i)close(analytic[i].v,(hi[i]-lo[i])/(2*epsilon),3e-7,"physical margin derivative");
    }
    auto changed=q;for(int i=0;i<ice_deviation_count;++i)changed[39+i]=.001*std::sin(double(i));
    // Keep the long-side width perturbation constant over each Lane.
    for(int row=0;row<4;++row)changed[39+52+row]=changed[39+48+row];
    IceToleranceGeometry<double> nonzero(board,changed.data());
    for(int row=0;row<4;++row){double upper[6],right[6],lower[6],cross[6];int kind;
        for(auto pair:{std::pair<int,double*>(300+3*row,upper),{301+3*row,right},{302+3*row,lower},{320+row,cross}})
            require(shared_entity_definition(board,6,changed.data(),changed.size(),pair.first,kind,pair.second,6,127)==Status::ok,"nonzero valid geometry");
        for(int axis=0;axis<3;++axis){close(upper[3+axis],right[axis],0.,"upper/right shared node");close(right[3+axis],lower[axis],0.,"right/lower shared node");}
        for(int end=0;end<2;++end){double* side=end?lower:upper;const double x=cross[3*end],y=cross[3*end+1];
            close((y-side[1])*(side[3]-side[0]),(x-side[0])*(side[4]-side[1]),1e-14,"cross endpoint on same side");}
    }
    int kind;double out[8];changed=q;changed[39+16]=.01;
    require(shared_entity_definition(board,6,changed.data(),changed.size(),100,kind,out,6,127)==Status::ok,"inclusive +1 percent");
    close(out[3],q[4]*q[5]*1.01,1e-15,"relative nominal radius");changed[39+16]=-.01;
    require(shared_entity_definition(board,6,changed.data(),changed.size(),100,kind,out,6,127)==Status::ok,"inclusive -1 percent");
    changed[39+16]=.0100001;require(shared_entity_definition(board,6,changed.data(),changed.size(),100,kind,out,6,127)==Status::outside_support,"reject tolerance overflow");
    changed=q;changed[5]=.98;changed[6]=.99;changed[7]=.999;changed[39+18]=.01;changed[39+19]=-.01;
    require(shared_entity_definition(board,6,changed.data(),changed.size(),100,kind,out,6,127)==Status::outside_support,"preserve nested ring order");
    changed=q;changed[7]=1/1.01;changed[39+18]=.01;
    require(shared_entity_definition(board,6,changed.data(),changed.size(),100,kind,out,6,127)==Status::outside_support,"reject coincident ring layers");
    changed=q;changed[14]=changed[1];
    require(shared_entity_definition(board,6,changed.data(),changed.size(),200,kind,out,6,127)==Status::outside_support,"connection must pass R strictly");
    changed=q;changed[14]=changed[8];
    require(shared_entity_definition(board,6,changed.data(),changed.size(),200,kind,out,6,127)==Status::outside_support,"connection must stop before N strictly");
    std::puts("ice tolerance: zero state, 107 geometry/margin derivatives, topology, order and bounds passed; no optimizer called");
}
