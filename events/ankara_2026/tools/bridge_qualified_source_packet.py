"""Bridge already-qualified point packets to the canonical source-update format.

This is provenance/format adaptation, not qualification. Original measurements,
seven point arrays, controls and seeds remain unchanged. The only default operator
alias is the declared saved whole-print q50 pass; unknown families are refused.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np

FIELDS={'pts','observation_ids','uv','kind','stations','source_rows','path_indices'}
ELIGIBILITY='External explicit path/family grant AND optional exact PTS/candidate grant AND saved source material/profile evidence; no residual gate'
GLYPH_FAMILY='whole_print_original_RGB_glyph_boundary'
GLYPH_PASS='exact_saved_key_numeric_pass_and_actual_source_physical_context'
GLYPH_OPERATOR='integrated_actual_signed_luminance_transition_q50'
DUPLICATE_ROW_SNAPSHOTS={'original_raw_row','original_row'}
DUPLICATE_FRAME_PAYLOADS={'rows','all_raw_readings','all_local_gate_diagnostics'}
DATA=Path('D:/CodexData/RinkSpace_Data')


def require(value,message):
    if not value:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def reference(path):
    path=Path(path).resolve(strict=True);return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def exact_row_sha(row):
    return hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def locate(value,folder):
    path=Path(value);return(path if path.is_absolute()else folder/path).resolve(strict=True)


def write_json_new(path,value):
    with Path(path).open('x',encoding='utf8',newline='\n')as stream:json.dump(value,stream,ensure_ascii=False,allow_nan=False,indent=2);stream.write('\n')


def bridge(job,parent_point_sha,config_ref):
    parent=Path(job['parent_packet']).resolve(strict=True);out=Path(job['output']).resolve()
    require(out.is_relative_to(DATA.resolve())and not out.exists()and not parent.is_relative_to(out),'Use a new separate Data packet directory')
    protected={}
    def protect(path,expected=None):
        pin=reference(path)
        if expected is not None:require(pin['sha256']==expected,'Input SHA differs: '+pin['path'])
        protected[pin['path']]=pin;return pin
    mapping_ref=protect(parent/'mapping.json',job['mapping_sha256']);mapping=read(mapping_ref['path'])
    require(mapping.get('independent_source_capture')is True and mapping.get('machine_qualification_only')is True
        and mapping.get('model_UV_or_source_PTS_generated')is False and mapping.get('source_eligibility')==ELIGIBILITY,
        'Parent must already have the exact explicit-grant AND saved-numeric source eligibility')
    require(mapping.get('pixel_domain')=='native_distorted'and mapping.get('image_size')==[1920,1080],'Parent native pixel domain differs')
    point_ref=protect(parent/'point_observations.npz',job['point_input_sha256'])
    require(mapping['point_input_sha256']==point_ref['sha256'],'Parent point mapping SHA differs')
    with np.load(point_ref['path'],allow_pickle=False)as values:points={key:values[key].copy()for key in values.files}
    require(set(points)==FIELDS,'Exactly seven unweighted point fields required')
    n=len(points['pts']);require(n==job['expected_rows']and n>0,'Parent point count differs')
    require(points['uv'].shape==(n,2)and all(value.shape==(n,)for key,value in points.items()if key!='uv'),'Point shapes differ')
    require(np.isfinite(points['uv']).all()and len(set(map(str,points['observation_ids'])))==n,'Invalid/duplicate original point rows')
    manifest_path=locate(mapping['source_row_manifest'],parent)
    manifest_ref=protect(manifest_path,job['source_row_manifest_sha256'])
    require(mapping['source_row_manifest_sha256']==manifest_ref['sha256'],'Parent source manifest SHA differs')
    qualification_sha=mapping['qualification_sha256'];require(re.fullmatch('[0-9a-f]{64}',qualification_sha)is not None,'Parent qualification SHA missing')
    locations={};groups=defaultdict(list);raw_files={}
    with manifest_path.open(encoding='utf-8-sig')as stream:
        for line_number,line in enumerate(stream,1):
            location=json.loads(line);i=location['packet_row']
            require(type(i)is int and 0<=i<n and i not in locations,'Parent row manifest must cover each packet row once')
            path=mapping['path_ids'][int(points['path_indices'][i])]
            require(location['observation_id']==str(points['observation_ids'][i])and location['source_row']==int(points['source_rows'][i])
                and location['pts']==int(points['pts'][i])and location['path_id']==path and np.array_equal(location['uv'],points['uv'][i]),'Seven-field packet identity differs from original source-row manifest')
            require(location.get('qualification_sha256')==qualification_sha and isinstance(location.get('existing_profile_material_pass'),str)
                and bool(location['existing_profile_material_pass'].strip()),'Every selected row needs its existing saved numerical/material pass')
            for key,array in (('kind','kind'),('station','stations')):
                if key in location:require(location[key]==points[array][i].item(),'Original manifest '+key+' differs')
            raw_path=locate(location['source_jsonl'],parent);raw_key=str(raw_path)
            if raw_key not in raw_files:raw_files[raw_key]=protect(raw_path,location['source_jsonl_sha256'])
            require(raw_files[raw_key]['sha256']==location['source_jsonl_sha256'],'Conflicting original capture file SHA')
            location=dict(location,source_jsonl=raw_key,parent_manifest_line=line_number)
            locations[i]=location;groups[raw_key,location['source_line']].append(location)
    require(len(locations)==n,'Parent row manifest is incomplete')
    qref=mapping.get('parent_qualified_source_rows')
    if qref is None:
        require(mapping.get('qualified_source_rows')and mapping.get('qualified_source_rows_sha256'),'Already-qualified source ledger reference required')
        qref=dict(path=str(locate(mapping['qualified_source_rows'],parent)),sha256=mapping['qualified_source_rows_sha256'])
    qualified_ref=protect(qref['path'],qref['sha256']);approved={};qualifiers={}
    with Path(qualified_ref['path']).open('rb')as stream:
        for line_number,raw in enumerate(stream,1):
            frame=json.loads(raw);origin=frame.get('original_capture_record',{})
            if not origin.get('source_jsonl'):continue
            key=(str(locate(origin['source_jsonl'],parent)),origin.get('source_line'))
            if key not in groups:continue
            require(frame.get('fit_eligible')is True,'Parent qualified frame was not already eligible')
            qualification=frame['qualification'];require(qualification['sha256']==qualification_sha,'Parent qualifier differs')
            if qualification['path']not in qualifiers:qualifiers[qualification['path']]=protect(qualification['path'],qualification_sha)
            for location in groups[key]:
                i=location['packet_row'];ri=location['row_index']
                require(i not in approved and type(ri)is int and 0<=ri<len(frame['rows']),'Already-qualified row locator is not unique')
                require(origin['source_jsonl_sha256']==location['source_jsonl_sha256']and origin['source_line_sha256']==location['source_line_sha256']
                    and frame['pts']==location['pts']and frame['source_BGR_sha256']==location['source_BGR_sha256'],'Parent qualified frame/raw source identity differs')
                approved[i]=dict(row_sha256=exact_row_sha(frame['rows'][ri]),qualified_ledger_line=line_number,
                    qualified_ledger_line_sha256=hashlib.sha256(raw).hexdigest())
    require(len(approved)==n,'Parent qualified ledger does not cover every selected raw row')
    copies={name:protect(parent/name)for name in ('controls.npy','shot_seed.json','pose_initialization.json')if(parent/name).is_file()}
    require('controls.npy'in copies and 'shot_seed.json'in copies,'Original controls and seed are required')
    out.mkdir(parents=True,exist_ok=False);new_capture=out/'qualified_source_rows.jsonl';new_locations={};alias_count=0;operator_preserved=0;removed_snapshots=0;frame_count=0
    with new_capture.open('xb')as destination:
        for source_path in sorted(raw_files):
            needed={number:items for(path,number),items in groups.items()if path==source_path}
            with Path(source_path).open('rb')as stream:
                for number,raw in enumerate(stream,1):
                    if number not in needed:continue
                    frame=json.loads(raw);selected=[];frame_locations=[];raw_line_sha=hashlib.sha256(raw).hexdigest()
                    for location in sorted(needed[number],key=lambda item:item['packet_row']):
                        i=location['packet_row'];ri=location['row_index']
                        require(raw_line_sha==location['source_line_sha256']and frame['pts']==int(points['pts'][i])
                            and frame['source_BGR_sha256']==location['source_BGR_sha256']and type(ri)is int and 0<=ri<len(frame['rows']),'Original source frame/row SHA, BGR or PTS differs')
                        row=frame['rows'][ri];require(row['path_id']==location['path_id']and np.array_equal(row['uv'],points['uv'][i])
                            and exact_row_sha(row)==approved[i]['row_sha256'],'Qualified parent row differs from its exact raw source measurement')
                        if 'pts'in row:require(row['pts']==int(points['pts'][i]),'Row source PTS differs')
                        kind=int(points['kind'][i]);station=float(points['stations'][i])
                        if 'kind'in row:require(row['kind']==kind,'Source kind differs')
                        if kind==0:require(row.get('station')==station,'Fixed source station differs')
                        elif 'station'in row:require(row['station']==station,'Original source station differs')
                        require(row.get('model_UV_synthesized')is not True and row.get('model_UV_used_as_measurements')is not True,'Synthetic source rows cannot be bridged')
                        original_locator={key:location[key]for key in ('source_jsonl','source_jsonl_sha256','source_line','source_line_sha256','row_index')}
                        original_locator.update(raw_row_sha256=approved[i]['row_sha256'],parent_packet_row=i,
                            parent_source_row_manifest_line=location['parent_manifest_line'])
                        derived={key:value for key,value in row.items()if key not in DUPLICATE_ROW_SNAPSHOTS}
                        removed_snapshots+=sum(key in row for key in DUPLICATE_ROW_SNAPSHOTS)
                        if 'measurement_operator'in row:
                            require(isinstance(row['measurement_operator'],str)and bool(row['measurement_operator'].strip()),'An existing empty operator may not be silently replaced')
                            operator_preserved+=1
                        else:
                            require(row.get('capture_family')==GLYPH_FAMILY and location['existing_profile_material_pass']==GLYPH_PASS,'No default measurement operator for this source family/pass')
                            derived['measurement_operator']=GLYPH_OPERATOR;alias_count+=1
                        derived.update(original_fit_eligible=row.get('fit_eligible'),fit_eligible=True,original_source_locator=original_locator,
                            source_qualification_reused=True,new_qualification_granted=False)
                        selected.append(derived);frame_locations.append((i,location,original_locator))
                    canonical={key:value for key,value in frame.items()if key not in DUPLICATE_FRAME_PAYLOADS}
                    canonical.update(rows=selected,fit_eligible=True,original_fit_eligible=frame.get('fit_eligible'),
                        original_source_locator=dict(source_jsonl=source_path,source_jsonl_sha256=raw_files[source_path]['sha256'],source_line=number,source_line_sha256=raw_line_sha),
                        parent_already_qualified_ledger=qualified_ref,parent_qualification_sha256=qualification_sha,
                        bridge_is_format_only=True,new_qualification_granted=False)
                    encoded=(json.dumps(canonical,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n').encode('utf8')
                    destination.write(encoded);frame_count+=1;new_line_sha=hashlib.sha256(encoded).hexdigest()
                    for new_index,(i,location,original_locator)in enumerate(frame_locations):
                        remapped=dict(location);remapped.pop('parent_manifest_line',None)
                        remapped.update(source_jsonl=str(new_capture),source_line=frame_count,source_line_sha256=new_line_sha,row_index=new_index,
                            original_source_locator=original_locator)
                        new_locations[i]=remapped
    require(len(new_locations)==n,'Canonical source omitted a packet row')
    capture_ref=reference(new_capture)
    new_manifest=out/'source_row_manifest.jsonl'
    with new_manifest.open('x',encoding='utf8',newline='\n')as stream:
        for i in range(n):
            row=new_locations[i];row['source_jsonl_sha256']=capture_ref['sha256']
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    for name,pin in dict(copies,**{'point_observations.npz':point_ref}).items():
        target=out/name;shutil.copyfile(pin['path'],target);require(sha(target)==pin['sha256'],'Byte-preserved packet copy differs: '+name)
    incoming=sorted({(int(pts),mapping['path_ids'][int(path)])for pts,path in zip(points['pts'],points['path_indices'])})
    replacement=dict(policy='explicit_parent_observation_ids',parent_point_input_sha256=parent_point_sha,
        groups=[dict(pts=pts,path_id=path,parent_observation_ids=[])for pts,path in incoming])
    result=copy.deepcopy(mapping);manifest_pin=reference(new_manifest)
    result.update(point_input_sha256=point_ref['sha256'],source_capture_jsonl='qualified_source_rows.jsonl',source_capture_sha256=capture_ref['sha256'],
        qualified_source_rows='qualified_source_rows.jsonl',qualified_source_rows_sha256=capture_ref['sha256'],source_row_manifest='source_row_manifest.jsonl',
        source_row_manifest_sha256=manifest_pin['sha256'],fit_eligible=True,source_replacement=replacement,
        source_packet_bridge=dict(parent_mapping=mapping_ref,parent_source_row_manifest=manifest_ref,parent_qualified_source_rows=qualified_ref,
            configuration=config_ref,implementation=reference(__file__),qualification_sha256=qualification_sha,new_qualification_granted=False,
            original_UV_PTS_quantiles_and_operators_unchanged=True,operator_alias_only_for_declared_saved_whole_print_pass=True))
    write_json_new(out/'mapping.json',result)
    for pin in protected.values():require(sha(pin['path'])==pin['sha256'],'Original input changed during bridge: '+pin['path'])
    receipt=dict(schema='rinkspace.qualified_source_packet_format_bridge.v1',parent_packet=str(parent),output=str(out),rows=n,frames=frame_count,
        seven_point_arrays_file_byte_identical=True,controls_and_seed_byte_identical=True,parent_qualification_reused=True,new_qualification_granted=False,
        measurement_operators_preserved=operator_preserved,declared_whole_print_operator_alias_rows=alias_count,
        duplicate_original_row_snapshots_replaced_by_locator=removed_snapshots,source_replacement=replacement,explicit_retired_parent_IDs=0,
        original_inputs=list(protected.values()),canonical_source=capture_ref,canonical_row_manifest=manifest_pin,
        output_point_file=reference(out/'point_observations.npz'),output_mapping=reference(out/'mapping.json'),
        RGB_or_profile_read=False,old_consumer_or_reader_called=False,build_prepare_or_fit_run=False,original_files_modified=False)
    write_json_new(out/'bridge_receipt.json',receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--bridge',action='store_true')
    args=parser.parse_args();config=read(args.config);require(re.fullmatch('[0-9a-f]{64}',config['parent_point_input_sha256'])is not None,'Parent point input SHA required')
    allowed={str((Path(job['parent_packet'])/'point_observations.npz').resolve()).lower()for job in config['jobs']}
    allowed|={str((Path(job['output'])/'point_observations.npz').resolve()).lower()for job in config['jobs']}
    def no_media(event,values):
        if event=='open'and isinstance(values[0],(str,bytes)):
            path=Path(values[0]);suffix=path.suffix.lower()
            if suffix in {'.png','.jpg','.jpeg','.bmp','.webp','.tif','.tiff','.mp4','.mkv','.mov','.avi'}or(suffix=='.npz'and str(path.resolve()).lower()not in allowed):
                raise RuntimeError('Format bridge may not read RGB, profiles or unlisted point arrays: '+str(path))
    sys.addaudithook(no_media)
    if not args.bridge:print(json.dumps(dict(schema='rinkspace.qualified_source_packet_bridge_plan.v1',configuration=config,write_executed=False),ensure_ascii=False));return
    config_ref=reference(args.config)
    for job in config['jobs']:
        receipt=bridge(job,config['parent_point_input_sha256'],config_ref)
        print(json.dumps(dict(output=receipt['output'],rows=receipt['rows'],operator_alias_rows=receipt['declared_whole_print_operator_alias_rows'],
            append_only=True,retired_parent_IDs=0,new_qualification_granted=False),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
