"""Generate a distinct source-audit reader without altering legacy frozen targets."""
from pathlib import Path
import argparse,hashlib,json

BASE_SHA256='802728c887971c6a80d9522bfa0597f1e3f63a2e5e3223b7daa0654edec46499'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def ref(path):return {'path':str(Path(path).resolve()),'sha256':sha(path)}

def generate_header(output):
    root=Path(__file__).resolve().parents[4]
    include=root/'calibration/rinkfit/cpp/include/rinkfit'
    base=include/'event_packet.hpp';layer=include/'source_audit_layer.inc'
    if sha(base)!=BASE_SHA256:raise ValueError('Legacy event reader changed; explicit compatibility review required')
    text=base.read_text(encoding='utf-8')
    text=text.replace('#include "rinkfit/event_io.hpp"','#include "rinkfit/event_io.hpp"\n#include "rinkfit/source_covariance.hpp"',1)
    marker='  eventio::Json source_exclusion,active_statistics;'
    if marker not in text:raise ValueError('EventPacket authority marker differs')
    text=text.replace(marker,marker+'\n  eventio::Json source_audit_revision;\n  std::string source_audit_revision_hash;\n  bool has_source_audit_revision=false;',1)
    marker='    if(semantics.contains("source_component_parent_manifest")'
    if marker not in text:raise ValueError('Semantic dispatch marker differs')
    text=text.replace(marker,'    if(semantics.contains("source_audit_parent_manifest")||semantics.contains("source_audit_revision")){load_source_audit_revision_layer();return;}\n'+marker,1)
    marker='  void load_source_component_recovery()'
    if marker not in text:raise ValueError('Semantic method marker differs')
    text=text.replace(marker,'#include "rinkfit/source_audit_layer.inc"\n'+marker,1)
    old='for(const auto& [name,desc]:manifest.at("assets").items())eventio::verify(root/name,desc.at("sha256"));'
    new='for(const auto& [name,desc]:manifest.at("assets").items())eventio::verify(root/std::filesystem::u8path(desc.value("file",name)),desc.at("sha256"));'
    if old not in text:raise ValueError('Packet asset resolution marker differs')
    text=text.replace(old,new,1)
    text='// Distinct audited-source reader. Legacy parent body SHA256 '+BASE_SHA256+' preserved.\n'+text
    out=Path(output);out.parent.mkdir(parents=True,exist_ok=True)
    encoded=text.encode('utf-8')
    if out.exists()and out.read_bytes()!=encoded:raise ValueError('Refuse replacing a frozen reader')
    out.write_bytes(encoded)
    manifest={'schema':'rinkfit.source-audit-cpp-reader-generation.v1','legacy_reader':ref(base),
        'source_audit_layer':ref(layer),'generator':ref(__file__),'generated_header':ref(out),
        'legacy_targets_untouched':True,'source_audit_is_not_direct_human_authority':True,
        'new_target_initialization_required':True}
    covariance=include/'source_covariance.hpp'
    if covariance.exists():manifest['covariance_validator']=ref(covariance)
    mpath=out.with_name('header_manifest.json')
    if mpath.exists()and json.loads(mpath.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Refuse replacing frozen reader generation manifest')
    mpath.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args(argv)
    print(json.dumps(generate_header(a.output)))
if __name__=='__main__':main()
