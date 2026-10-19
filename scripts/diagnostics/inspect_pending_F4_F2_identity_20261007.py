"""Read exact original exposures for two pending identity proposals.

Produces inspection images only; source rows, q, manifest, and HELD stay intact.
In particular native 231012864 belongs to the final close shot, whereas
231013120 is already a different shot and cannot supply temporal correspondence.
"""
from pathlib import Path
import hashlib
import json
import pickle
import av
import cv2

BASE = Path('D:/CodexData/RinkSpace_Data/events/ankara_2026')
OUT = BASE/'diagnostics/external_expert_20261007/all_original_absolute_source_audit/native_rgb_review/pending_F4_F2_identity'
PACKET = BASE/'calibration/runs/solver_acceleration_20261003/shared_board_course_candidate/actual_absolute_source_material_packet/actual_absolute_FIT_HELD_source_material_rows_compact.pkl'
PTS = [230967808, 230989312, 231006208, 231009280, 231012608,
       231012864, 229342976, 229352192, 229361408]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    packet_bytes = PACKET.read_bytes()
    rows = pickle.loads(packet_bytes)['programme_absolute_rows']
    evidence = []
    with av.open(str(BASE/'media/originals/oxXu1avhrvs.mp4')) as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        for pts in PTS:
            container.seek(pts-15360, stream=stream, backward=True, any_frame=False)
            frame = next(f for f in container.decode(stream) if f.pts == pts)
            bgr = frame.to_ndarray(format='bgr24')
            path = OUT/f'raw_native_{pts}.png'
            cv2.imwrite(str(path), bgr)
            evidence.append(dict(native_pts=pts, path=str(path),
                file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decoded_BGR_sha256=hashlib.sha256(bgr.tobytes()).hexdigest()))
    proposals = []
    for number, actual, native, box in [
        ('F4_outer', 231012608, 231012864, [40, 750, 730, 890]),
        ('F2_outer', 229352192, 229352192, [40, 680, 360, 810])]:
        record = next(r for r in rows if r['source_row'].get('id') == number
                      and r['source_row'].get('pts') == actual)
        source = record['source_row']
        assert source.get('source_native_pts', actual) == native
        assert record['collection_role'] == 'HELD'
        assert source['split'] == 'unfitted_identity_pending'
        bgr = cv2.imread(str(OUT/f'raw_native_{native}.png'))
        marked = bgr.copy()
        for i, (u, v) in enumerate(source['uv']):
            point = (int(round(u)), int(round(v)))
            cv2.circle(marked, point, 3, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(marked, str(i), (point[0]+3, point[1]-3),
                        cv2.FONT_HERSHEY_SIMPLEX, .34, (255, 0, 255), 1, cv2.LINE_AA)
        x0, y0, x1, y1 = box
        for image, suffix in [(bgr, 'RGB'), (marked, 'stored_UV')]:
            cv2.imwrite(str(OUT/f'{number}_native_{native}_{suffix}_crop.png'),
                        image[y0:y1, x0:x1])
        proposals.append(dict(id=number, actual_pts=actual, source_native_pts=native,
            collection_role=record['collection_role'], source_split=source['split'],
            unchanged_original_UV=source['uv'], crop_xyxy=box,
            origin=source.get('origin'), authority=source.get('source_authority_admission'),
            physical_identity_review='pending original RGB review'))
    assert PACKET.read_bytes() == packet_bytes
    report = dict(schema='rinkspace.pending-F4-F2-original-RGB-inspection.v1',
        projection_used=False, source_or_model_modified=False,
        original_packet_sha256=hashlib.sha256(packet_bytes).hexdigest(),
        source_roles_and_UV_unchanged=True, evidence=evidence, proposals=proposals,
        images_are_samples_not_complete_shot_accuracy_validation=True)
    (OUT/'decode_provenance.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2,
                   default=lambda value: value.tolist())+'\n', encoding='utf8')
    print(json.dumps(dict(out=str(OUT), decoded=PTS)))


if __name__ == '__main__':
    main()
