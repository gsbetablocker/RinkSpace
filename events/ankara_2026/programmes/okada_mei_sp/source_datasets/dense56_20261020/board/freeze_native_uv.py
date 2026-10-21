from source_reader_anchored import *
def frozen_payload(q):return {ro:{k:z[k]for k in ['full_native_y','full_direct','full_gradient','full_chroma_before','full_chroma_after']}for ro,z in q['roles'].items()}|{'blue_cap_lower_y':q['roles']['board.top']['material_band']['full_blue_cap_lower_y']}
ordered=hashlib.sha256()
with gzip.open(O/'frozen_native_uv.jsonl.gz','wt',encoding='utf8')as f:
 for p in PTS:
  q=readj(O/'anchored_frames'/(str(p)+'.json'));v=hashlib.sha256(json.dumps(frozen_payload(q),separators=(',',':'),allow_nan=False).encode()).hexdigest();ordered.update((str(p)+':'+v+'\n').encode());f.write(json.dumps(dict(native_pts=p,decoded_native_BGR_sha256=q['decoded_native_BGR_sha256'],frozen_native_material_arrays_sha256=v),separators=(',',':'))+'\n')
write(O/'material_uv_freeze.json',dict(complete=True,native_frames=1862,reader_sha256=sha(O/'source_reader_anchored.py'),arrays=['full_native_y','full_direct','full_gradient','full_chroma_before','full_chroma_after','blue_cap_lower_y'],per_frame_receipt_file=str(O/'frozen_native_uv.jsonl.gz'),per_frame_receipt_sha256=sha(O/'frozen_native_uv.jsonl.gz'),ordered_frozen_native_material_arrays_sha256=ordered.hexdigest(),final_mask_qualification_can_change_eligibility_metadata_only=True,source_UV_may_not_move=True,fitting_started=False))
print(json.dumps(readj(O/'material_uv_freeze.json')),flush=True)
