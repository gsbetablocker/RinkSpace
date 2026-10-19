#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace rinkfit::eventio {
using Json = nlohmann::json;
inline Json read_json(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary); if (!in) throw std::runtime_error("cannot read " + path.string());
  Json value; in >> value; return value;
}
inline std::vector<std::uint8_t> bytes(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary | std::ios::ate); if (!in) throw std::runtime_error("cannot read " + p.string());
  const auto n = in.tellg(); if (n < 0) throw std::runtime_error("cannot size " + p.string());
  std::vector<std::uint8_t> out(static_cast<std::size_t>(n)); in.seekg(0);
  if (!out.empty() && !in.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()))) throw std::runtime_error("short read " + p.string());
  return out;
}
class SHA256Stream {
 std::array<std::uint32_t,8> H{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
 std::array<std::uint8_t,64> pending{};std::size_t used=0;std::uint64_t total=0;
 void compress(const std::uint8_t*data){
  constexpr std::uint32_t K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

  auto r=[](std::uint32_t x,int n){return (x>>n)|(x<<(32-n));};
  std::uint32_t w[64]; for(int i=0;i<16;++i)w[i]=(std::uint32_t(data[4*i])<<24)|(std::uint32_t(data[4*i+1])<<16)|(std::uint32_t(data[4*i+2])<<8)|data[4*i+3];
   for(int i=16;i<64;++i){auto a=w[i-15],b=w[i-2];w[i]=w[i-16]+(r(a,7)^r(a,18)^(a>>3))+w[i-7]+(r(b,17)^r(b,19)^(b>>10));}
   auto a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
   for(int i=0;i<64;++i){auto t1=h+(r(e,6)^r(e,11)^r(e,25))+((e&f)^((~e)&g))+K[i]+w[i];auto t2=(r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
   H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;
 }
 public:
 void update(const std::uint8_t*data,std::size_t size){total+=size;while(size){const auto n=std::min(size,pending.size()-used);std::memcpy(pending.data()+used,data,n);used+=n;data+=n;size-=n;if(used==pending.size()){compress(pending.data());used=0;}}}
 std::string finish(){const auto bits=total*8;std::array<std::uint8_t,128>pad{};pad[0]=128;const auto padding=used<56?56-used:120-used;update(pad.data(),padding);std::array<std::uint8_t,8>length{};for(int i=0;i<8;++i)length[7-i]=static_cast<std::uint8_t>(bits>>(8*i));update(length.data(),length.size());std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto h:H)out<<std::setw(8)<<h;return out.str();}
};
inline std::string sha256(const std::vector<std::uint8_t>&data){SHA256Stream h;h.update(data.data(),data.size());return h.finish();}
inline std::string file_sha256(const std::filesystem::path&p){std::ifstream in(p,std::ios::binary);if(!in)throw std::runtime_error("cannot read "+p.string());SHA256Stream h;std::vector<std::uint8_t>buffer(1024*1024);while(in){in.read(reinterpret_cast<char*>(buffer.data()),static_cast<std::streamsize>(buffer.size()));const auto n=in.gcount();if(n>0)h.update(buffer.data(),static_cast<std::size_t>(n));}if(!in.eof())throw std::runtime_error("short hash read "+p.string());return h.finish();}
inline void verify(const std::filesystem::path& p,const std::string& expected){if(file_sha256(p)!=expected)throw std::runtime_error("SHA256 mismatch: "+p.string());}
template<class T> std::vector<T> read_array(const std::filesystem::path& root,const Json& d,const std::string& dtype) {
  if(d.at("dtype")!=dtype || d.value("byte_order",std::string("little"))!="little" || d.value("layout",std::string("C"))!="C")throw std::runtime_error("binary ABI dtype/layout/endian mismatch");
  auto p=root/d.at("file").get<std::string>(); auto b=bytes(p); if(sha256(b)!=d.at("sha256").get<std::string>())throw std::runtime_error("binary SHA256 mismatch: "+p.string());
  std::size_t count=1;for(const auto& dim:d.at("shape"))count*=dim.get<std::size_t>();if(b.size()!=count*sizeof(T))throw std::runtime_error("binary byte/shape mismatch: "+p.string());
  const std::uint16_t endian=1;if(*reinterpret_cast<const std::uint8_t*>(&endian)!=1)throw std::runtime_error("reader requires little-endian host");
  std::vector<T> out(count);if(!b.empty())std::memcpy(out.data(),b.data(),b.size());return out;
}
inline void atomic_json(const std::filesystem::path& path,const Json& value){const auto tmp=path.string()+".tmp";{std::ofstream out(tmp,std::ios::binary|std::ios::trunc);if(!out)throw std::runtime_error("cannot write "+tmp);out<<value.dump(2)<<'\n';out.flush();if(!out)throw std::runtime_error("cannot flush "+tmp);}std::error_code ec;std::filesystem::rename(tmp,path,ec);if(ec){std::filesystem::remove(path,ec);std::filesystem::rename(tmp,path);}}
} // namespace rinkfit::eventio
