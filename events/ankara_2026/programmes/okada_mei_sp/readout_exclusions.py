"""Verify a complete source-bound exclusion index without changing raw arrays."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bound_path(value,parent):
    path=Path(value)
    return (path if path.is_absolute() else parent/path).resolve()


def retained_runs(candidate_ids,excluded):
    """Keep original order and break at every denied actual readout."""
    result=[];run=[]
    for candidate in candidate_ids:
        if int(candidate) in excluded:
            if run:result.append(run);run=[]
        else:run.append(int(candidate))
    if run:result.append(run)
    return result


class ReadoutExclusions:
    def __init__(self,coverage):
        self.path=Path(coverage).resolve();self.sha256=digest(self.path)
        report=json.loads(self.path.read_bytes());self.report=report
        if (report.get('schema')!='rinkspace.all_raw_candidates_added_person_guard.v1' or
            report.get('status')!='complete' or report.get('all_raw_candidates_considered') is not True or
            report.get('source_candidate_id_spaces_separate') is not True):
            raise ValueError('Complete all-original-candidate source exclusion coverage required')
        self.index=report['index'];path=bound_path(self.index['path'],self.path.parent)
        if digest(path)!=self.index['sha256']:raise ValueError('Readout exclusion full-exposure index SHA changed')
        self.frames={};counts=Counter();scopes={}
        data=gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()
        for text in data.decode('utf8').splitlines():
            if not text.strip():continue
            row=json.loads(text);key=(row['capture'],row['shot'],int(row['pts']))
            if key in self.frames:raise ValueError('Duplicate exclusion capture/shot/PTS')
            if row.get('applied_to_all_original_candidate_ids') is not True:
                raise ValueError('Shape-selected-only exclusions cannot guard raw candidate propagation')
            self.frames[key]=row;counts[key[0]]+=1;scopes.setdefault(key[0],set()).add(key[1:])
        if (len(self.frames)!=report['output_frames'] or set(counts)!=set(report['capture_bindings']) or
            any(n!=report['source_exposures_per_capture']for n in counts.values()) or
            any(scope!=next(iter(scopes.values()))for scope in scopes.values())):
            raise ValueError('Incomplete exclusion full-exposure denominator')
        by_shot=Counter(capture+'/'+shot for capture,shot,pts in self.frames)
        if any(by_shot.get(key)!=value['frames']for key,value in report['by_capture_shot'].items()) or set(by_shot)!=set(report['by_capture_shot']):
            raise ValueError('Exclusion shot denominator differs')
        for capture,reference in report['capture_bindings'].items():
            binding=bound_path(reference['path'],self.path.parent)
            if digest(binding)!=reference['sha256']:raise ValueError('Readout exclusion capture binding SHA changed')
        self.bound={};self.checked={}

    def bind_capture(self,name,capture_path):
        reference=self.report['capture_bindings'].get(name)
        if reference is None:raise ValueError('Missing exclusion capture candidate-ID space')
        actual=Path(capture_path).resolve()/'binding.json'
        if actual!=bound_path(reference['path'],self.path.parent) or digest(actual)!=reference['sha256']:
            raise ValueError('Readout exclusions belong to a different capture candidate-ID space')
        self.bound[name]=actual.parent

    def for_frame(self,capture,shot,pts,raw_path,raw_sha256,source_BGR_sha256,source_RGB_sha256,candidate_count):
        key=(capture,shot,int(pts));row=self.frames.get(key)
        if row is None:raise ValueError('Exclusion exposure missing from complete index')
        if capture not in self.bound:raise ValueError('Exclusion capture must be explicitly bound')
        raw_path=Path(raw_path).resolve();reference=row['raw_measurement']
        if (raw_path!=bound_path(reference['path'],self.path.parent) or raw_sha256!=reference['sha256'] or
            source_BGR_sha256!=row['source_BGR_sha256'] or source_RGB_sha256!=row['source_RGB_sha256']):
            raise ValueError('Readout exclusion raw measurement/RGB binding differs')
        if raw_path.parent.parent!=self.bound[capture] or raw_path.parent.name!=shot or raw_path.stem!=str(pts):
            raise ValueError('Readout exclusion raw exposure path differs')
        arrays=bound_path(row['arrays']['path'],self.path.parent)
        def stamp(path):
            value=path.stat()
            return (value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
        signature=(stamp(raw_path),stamp(arrays),int(candidate_count))
        cached=self.checked.get(key)
        if cached is not None and cached[0]==signature:return cached[1]
        if digest(raw_path)!=reference['sha256'] or digest(arrays)!=row['arrays']['sha256']:
            raise ValueError('Readout exclusion raw/ID NPZ SHA changed')
        with np.load(arrays,allow_pickle=False)as z:
            ids=z['affected_candidate_ids'];bits=z['trigger_bits']
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or bits.shape!=ids.shape or bits.dtype!=np.uint8 or
            len(ids)!=row['affected_candidate_count'] or (len(ids)>1 and not(np.diff(ids.astype(np.int64))>0).all()) or
            (len(ids)and((ids<0).any()or(ids>=candidate_count).any()or(bits==0).any()))):
            raise ValueError('Invalid original exclusion candidate IDs/bits/bounds')
        population=row.get('evaluated_original_candidate_count')
        if population is not None and population!=candidate_count:raise ValueError('Exclusion original candidate denominator changed')
        if (not row['unresolved_support'] or row['added_source_exclusion_pixels']==0)and len(ids):
            raise ValueError('Normal/zero-added-guard exposure exclusions must be empty')
        result=frozenset(map(int,ids));self.checked[key]=(signature,result)
        return result

    def receipt(self):
        return dict(coverage=dict(path=str(self.path),sha256=self.sha256),index=self.index,
            helper_sha256=digest(__file__),full_exposure_index_verified=True,capture_bindings_verified=True,
            verified_consumed_exposures=len(self.checked),verified_excluded_candidates=sum(len(value[1])for value in self.checked.values()))
