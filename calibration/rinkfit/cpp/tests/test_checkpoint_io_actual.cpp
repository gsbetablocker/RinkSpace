#include "rinkfit/augmented_lagrangian.hpp"
#include "rinkfit/state_text_codec.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <cstring>
#include <locale>
int main(int argc,char**argv){try{using namespace rinkfit;for(std::uint64_t k=0,h=1;k<10000;++k){h=h*6364136223846793005ULL+1442695040888963407ULL;double v;std::memcpy(&v,&h,8);if(k==0)v=0.;if(k==1)v=-0.;if(k==2)v=std::numeric_limits<double>::denorm_min();if(!std::isfinite(v))continue;std::ostringstream ref;ref.imbue(std::locale::classic());ref<<std::hexfloat<<v;state_text::Writer fast;fast.number(v,true);if(ref.str()!=fast.text)throw std::runtime_error("hex formatting differs: "+ref.str()+" / "+fast.text);}auto t0=std::chrono::steady_clock::now();std::ifstream in(argv[1]);auto c=read_checkpoint(in);auto t1=std::chrono::steady_clock::now();{std::ofstream out(argv[2]);write_checkpoint(out,c);}auto t2=std::chrono::steady_clock::now();std::ifstream back(argv[2]);auto d=read_checkpoint(back);auto t3=std::chrono::steady_clock::now();if(c.state.token()!=d.state.token())throw std::runtime_error("round trip token changed");const auto&a=c.state.data().parameters;const auto&b=d.state.data().parameters;if(a.size()!=b.size())throw std::runtime_error("count changed");for(std::size_t i=0;i<a.size();++i)if(a[i].name!=b[i].name||a[i].equality_auxiliary!=b[i].equality_auxiliary||a[i].values.size()!=b[i].values.size()||std::memcmp(a[i].values.data(),b[i].values.data(),a[i].values.size()*8))throw std::runtime_error("physical bit pattern changed");auto sec=[](auto a,auto b){return std::chrono::duration<double>(b-a).count();};std::cout<<"blocks="<<a.size()<<" token="<<c.state.token()<<" read_seconds="<<sec(t0,t1)<<" write_seconds="<<sec(t1,t2)<<" reread_seconds="<<sec(t2,t3)<<" identical_bits=true\n";return 0;}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
