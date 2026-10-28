"""The shared connected review, bound to S02's immutable source/native snapshot."""
import sys
import points_s02 as p
import review_points_s04 as shared

shared.OUT=p.OUT
shared.ADS=p.OUT/'ads'
shared.FIRST,shared.LAST,shared.STEP=p.FIRST,p.LAST,p.STEP
shared.SHOT_LABEL='S02'
shared.CurrentNativeGrid=p.NativeGrid

if __name__=='__main__':
    # SourceReview reads the common evidence layout used by existing S04.
    (shared.ADS/'evidence').mkdir(exist_ok=True)
    (shared.ADS/'evidence/visibility.json').write_bytes((p.OUT/'evidence/visibility.json').read_bytes())
    shared.main(['--input-dir',str(p.OUT/'points'),*sys.argv[1:]])
