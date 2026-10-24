#pragma once
#include <nlohmann/json.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <map>
#include <memory>

namespace rinkfit::eventio {
using Json = nlohmann::json;
// Data-only immutable source snapshot. Enabled only during full cold dataset validation.
struct SnapshotBlob { std::vector<std::uint8_t> bytes; std::string hash; };
inline bool snapshot_capture_enabled=false;
inline std::map<std::string,SnapshotBlob> snapshot_blobs;
inline std::string snapshot_key(const std::filesystem::path& p){return std::filesystem::absolute(p).lexically_normal().u8string();}
inline std::vector<std::uint8_t> disk_bytes(const std::filesystem::path& p){
 std::ifstream in(p,std::ios::binary|std::ios::ate);if(!in)throw std::runtime_error("cannot read "+p.u8string());auto n=in.tellg();if(n<0)throw std::runtime_error("cannot size source");std::vector<std::uint8_t> b(static_cast<std::size_t>(n));in.seekg(0);if(!b.empty()&&!in.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size())))throw std::runtime_error("short source read");return b;
}
inline std::vector<std::uint8_t> bytes(const std::filesystem::path& p){
 if(!snapshot_capture_enabled)return disk_bytes(p);
 const auto k=snapshot_key(p);auto it=snapshot_blobs.find(k);if(it==snapshot_blobs.end())it=snapshot_blobs.emplace(k,SnapshotBlob{disk_bytes(p),{}}).first;return it->second.bytes;
}
inline std::string snapshot_text(const std::filesystem::path& p){const auto b=bytes(p);return std::string(b.begin(),b.end());}
inline Json read_json(const std::filesystem::path& p){const auto b=bytes(p);return Json::parse(b.begin(),b.end());}
inline std::string sha256(const std::vector<std::uint8_t>& data) {
  constexpr std::uint32_t K[64]={0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
  auto r=[](std::uint32_t x,int n){return (x>>n)|(x<<(32-n));};
  std::array<std::uint32_t,8> H{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
  auto v=data; const std::uint64_t bits=static_cast<std::uint64_t>(v.size())*8; v.push_back(128);
  while(v.size()%64!=56) v.push_back(0); for(int i=7;i>=0;--i)v.push_back(static_cast<std::uint8_t>(bits>>(8*i)));
  for(std::size_t off=0;off<v.size();off+=64){std::uint32_t w[64]; for(int i=0;i<16;++i)w[i]=(std::uint32_t(v[off+4*i])<<24)|(std::uint32_t(v[off+4*i+1])<<16)|(std::uint32_t(v[off+4*i+2])<<8)|v[off+4*i+3];
   for(int i=16;i<64;++i){auto a=w[i-15],b=w[i-2];w[i]=w[i-16]+(r(a,7)^r(a,18)^(a>>3))+w[i-7]+(r(b,17)^r(b,19)^(b>>10));}
   auto a=H[0],b=H[1],c=H[2],d=H[3],e=H[4],f=H[5],g=H[6],h=H[7];
   for(int i=0;i<64;++i){auto t1=h+(r(e,6)^r(e,11)^r(e,25))+((e&f)^((~e)&g))+K[i]+w[i];auto t2=(r(a,2)^r(a,13)^r(a,22))+((a&b)^(a&c)^(b&c));h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
   H[0]+=a;H[1]+=b;H[2]+=c;H[3]+=d;H[4]+=e;H[5]+=f;H[6]+=g;H[7]+=h;}
  std::ostringstream out;out<<std::hex<<std::setfill('0');for(auto h:H)out<<std::setw(8)<<h;return out.str();
}
inline std::string file_sha256(const std::filesystem::path& p) {
 if(!snapshot_capture_enabled)return sha256(bytes(p));
 const auto k=snapshot_key(p);if(!snapshot_blobs.count(k))snapshot_blobs.emplace(k,SnapshotBlob{disk_bytes(p),{}});auto& b=snapshot_blobs.at(k);if(b.hash.empty())b.hash=sha256(b.bytes);return b.hash;
}
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
