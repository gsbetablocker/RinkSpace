"""Record exact-clock full videos alongside their programme candidate bindings."""
from pathlib import Path
from hashlib import sha256
import json

from assemble_full import FILES, TARGET
from joint_inputs import pin, read, check
from rinkspace.path_policy import guard_project_write

EDITIONS={'jin':'jin_shuxian','okada':'okada_mei'}


def main():
    planned=[]
    scene_digest=None
    for name,path in FILES.items():
        binding=read(path)
        path=guard_project_write(path)
        before=pin(path)
        receipt_path=TARGET/(EDITIONS[name]+'_receipt.json')
        receipt=read(receipt_path)
        binding_key='binding_before_full_review_registration'
        old_key='binding'
        assert receipt.get(binding_key,receipt.get(old_key))==before
        if old_key in receipt:
            receipt[binding_key]=receipt.pop(old_key)
            temp=receipt_path.with_suffix('.staged.json')
            with temp.open('x',encoding='utf8') as f:
                json.dump(receipt,f,indent=2,ensure_ascii=False,allow_nan=False)
                f.write('\n')
            temp.replace(receipt_path)
        assert binding['programme_id']==receipt['programme']
        assert binding['geometry']==receipt['scene']
        digest=receipt['scene']['value_sha256']
        if scene_digest is not None and digest!=scene_digest:
            raise ValueError('Programme bindings disagree about the shared scene')
        scene_digest=digest
        assert receipt['validation']['frame_pts_exact']
        assert receipt['validation']['decoded_frames']==binding['source_native_PTS']['exposures']
        check(receipt['video'])
        binding['full_review']={
            'video':receipt['video'],
            'receipt':pin(receipt_path),
            'original_source_seconds':[receipt['source_first_seconds'],receipt['source_last_seconds']],
            'frames':receipt['validation']['decoded_frames'],
            'frame_rate':receipt['frame_rate'],
            'exact_source_PTS':True,
            'audio_preserved_from_source':bool(receipt['output_audio']),
        }
        staged=guard_project_write(path.with_suffix('.staged.json'))
        with staged.open('x',encoding='utf8') as f:
            json.dump(binding,f,indent=2,ensure_ascii=False,allow_nan=False)
            f.write('\n')
        assert read(staged)==binding
        planned.append((before,path,staged))
    for before,path,staged in planned:
        check(before)
        staged.replace(path)
        a=read(path)['full_review']
        print(path.name,a['frames'],a['video']['path'])


if __name__=='__main__':
    main()
