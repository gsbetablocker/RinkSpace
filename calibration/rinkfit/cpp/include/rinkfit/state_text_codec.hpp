#pragma once
#include <charconv>
#include <cmath>
#include <cstdint>
#include <istream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
namespace rinkfit::state_text {
inline void require(bool ok,const char* reason){if(!ok)throw std::invalid_argument(reason);}
struct Writer {
 std::string text;
 void raw(std::string_view v){text.append(v.data(),v.size());}
 void ch(char c){text.push_back(c);}
 void quoted(std::string_view v){ch('"');for(char c:v){if(c=='"'||c=='\\')ch('\\');ch(c);}ch('"');}
 template<class T>void integer(T v){char s[32];auto r=std::to_chars(s,s+sizeof(s),v);require(r.ec==std::errc(),"checkpoint integer formatting failed");raw({s,std::size_t(r.ptr-s)});}
 void number(double v,bool hexadecimal=false){char s[128];auto r=hexadecimal?std::to_chars(s,s+sizeof(s),v,std::chars_format::hex,13):std::to_chars(s,s+sizeof(s),v,std::chars_format::general,17);require(r.ec==std::errc(),"checkpoint number formatting failed");std::string_view z(s,std::size_t(r.ptr-s));if(hexadecimal&&std::isfinite(v)){if(!z.empty()&&z.front()=='-'){ch('-');z.remove_prefix(1);}raw("0x");}raw(z);}
};
struct Reader {
 std::string text;std::size_t pos=0;
 explicit Reader(std::istream& in):text(std::istreambuf_iterator<char>(in),{}){require(!in.bad(),"checkpoint read failed");}
 void space(){while(pos<text.size()&&(text[pos]==' '||text[pos]=='\n'||text[pos]=='\r'||text[pos]=='\t'))++pos;}
 std::string_view token(){space();auto start=pos;while(pos<text.size()&&text[pos]!=' '&&text[pos]!='\n'&&text[pos]!='\r'&&text[pos]!='\t')++pos;require(pos>start,"truncated checkpoint token");return {text.data()+start,pos-start};}
 std::string quoted(){space();require(pos<text.size()&&text[pos++]=='"',"checkpoint quoted string expected");std::string out;auto start=pos;while(pos<text.size()){char c=text[pos++];if(c=='"'){out.append(text,start,pos-start-1);return out;}if(c=='\\'){out.append(text,start,pos-start-1);require(pos<text.size(),"truncated checkpoint escape");out.push_back(text[pos++]);start=pos;}}throw std::invalid_argument("truncated checkpoint string");}
 std::size_t count(){auto z=token();std::uint64_t n=0;auto r=std::from_chars(z.data(),z.data()+z.size(),n);require(r.ec==std::errc()&&r.ptr==z.data()+z.size()&&n<=20000000,"checkpoint count exceeds declared parser limit");return std::size_t(n);}
 double number(){auto z=token();double n=0;auto r=std::from_chars(z.data(),z.data()+z.size(),n,std::chars_format::general);require(r.ec==std::errc()&&r.ptr==z.data()+z.size()&&!std::isnan(n),"invalid checkpoint number");return n;}
};
}
