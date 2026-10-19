import numpy as np
from calibration.program_tracking.compact_source_json import pack_source_json
from calibration.program_tracking.gray_source_review_pack_headers_v2 import read_gray_source_review_headers


def test_shared_original_uncertainty_scalars_and_optional_query_locator(tmp_path):
    native=dict(quantization_step_px=1.,source_interpolation_grid_px=1/32.,
        original_long_evidence='source proof retained in the input pack')
    board=dict(source_native_pixel_quantization_step_px=1.,interpolation_grid_px=1/32.)
    rows=[]
    for i,unc in enumerate((native,native,board)):
        row=dict(shot_id='cut',actual_pts=100+i,native_pts=100+i,scope='ice',
            physical_material_id='F1',model_target=dict(curve_id='F1'),split='fit',
            point_count=1,source_readout_uncertainty=unc)
        if i==1:row['source_query_indices']=[42]
        rows.append(row)
    path=tmp_path/'source.npz'
    np.savez(path,rows_json=pack_source_json(rows),row_offsets=np.arange(4,dtype=np.int64))
    read,offsets=read_gray_source_review_headers(path)
    assert np.array_equal(offsets,np.arange(4))
    assert read[0]['source_readout_uncertainty']==read[1]['source_readout_uncertainty']==dict(
        quantization_step_px=1.,source_interpolation_grid_px=1/32.)
    assert read[2]['source_readout_uncertainty']==board
    assert 'source_query_indices' not in read[0]
    assert read[1]['source_query_indices']==[42]
