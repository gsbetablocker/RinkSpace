"""Value-level failure injection for the new immutable current observations."""
import dataclasses as D
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from rinksolve import schema as S
from rinksolve.observations import (validate_batches, observation_id, split_observation_id,
                                   load_import, validate_current_support, CLOCK_ID)

DATA = Path(r"D:/CodexData/RinkSpace_Data")


class CurrentFixture:
    def __init__(self, root):
        self.root, self.serial = Path(root), 0
        self.refs, self.tables, self.indices = [], [], []
        self.scalar = S.UnitRef(S.Unit.DIMENSIONLESS)
        frames = ["O6j8L8TmfKo:256256","O6j8L8TmfKo:256512"]
        self.frames = tuple(S.SourceFrame(fid,CLOCK_ID,p,1920,1080,"source")
                            for fid,p in zip(frames,(256256,256512)))
        self.table("frames",{"frame_id":(np.asarray(frames,dtype="S32"),S.ColumnMeaning.STABLE_ID,self.scalar)},
                   "frame_id")
        self.table("raw",{
            "observation_id":(np.asarray([observation_id("dataset",p,"line/shared-id") for p in (256256,256512)],dtype="S64"),S.ColumnMeaning.STABLE_ID,self.scalar),
            "source_id":(np.asarray(["line/shared-id"]*2,dtype="S32"),S.ColumnMeaning.OTHER,self.scalar),
            "source_dataset_id":(np.asarray(["dataset"]*2,dtype="S16"),S.ColumnMeaning.OTHER,self.scalar),
            "native_pts":(np.asarray([256256,256512],dtype="<i8"),S.ColumnMeaning.NATIVE_PTS,S.UnitRef(S.Unit.NATIVE_PTS,CLOCK_ID)),
            "frame_id":(np.asarray(frames,dtype="S32"),S.ColumnMeaning.SOURCE_FRAME_ID,self.scalar),
            "u":(np.asarray([0.5,1919.0],dtype="<f8"),S.ColumnMeaning.RAW_U,S.UnitRef(S.Unit.PIXEL)),
            "v":(np.asarray([100.0,1000.0],dtype="<f8"),S.ColumnMeaning.RAW_V,S.UnitRef(S.Unit.PIXEL))},
            "observation_id",(S.ColumnForeignKey("frame_id","frames","frame_id"),))
        self.selection = S.SelectionRef("selected","raw",1,S.SelectionMode.SMALL_IDS,
            ids=tuple(x.decode() for x in np.load(self.refs_by_name()["raw.observation_id"].path)),selected_count=2)

    def save(self, aid, a, unit, role=S.ArrayRole.IMMUTABLE_EVIDENCE):
        self.serial += 1
        path=self.root/(str(self.serial)+".npy")
        np.save(path,a,allow_pickle=False)
        ref=S.TypedArrayRef(aid,str(path),"npy",a.dtype.str,a.shape,unit,role,True,1,
                            clock_id=CLOCK_ID if unit.unit is S.Unit.NATIVE_PTS else None)
        self.refs.append(ref)
        return aid

    def table(self,tid,values,primary,fkeys=()):
        cols=tuple(S.ColumnRef(name,self.save(tid+"."+name,a,unit),meaning,"test current evidence")
                   for name,(a,meaning,unit) in values.items())
        key=values[primary][0]
        order=np.argsort(key).astype("<u8")
        ix=S.IdIndexRef("index."+tid,tid,self.save(tid+".keys",key[order],self.scalar,S.ArrayRole.SEMANTIC_FOREIGN_KEYS),
                         self.save(tid+".rows",order,self.scalar,S.ArrayRole.SEMANTIC_FOREIGN_KEYS),1,"sorted exact")
        self.indices.append(ix)
        self.tables.append(S.ColumnTableRef(tid,len(key),1,S.ArrayRole.IMMUTABLE_EVIDENCE,cols,primary,ix.id,tuple(fkeys),("source",)))

    def refs_by_name(self):
        return {a.id:a for a in self.refs}

    def replace_values(self, aid, values):
        old=self.refs_by_name()[aid]
        self.serial+=1
        path=self.root/(str(self.serial)+".npy")
        np.save(path,np.asarray(values,dtype=old.dtype),allow_pickle=False)
        self.refs=[D.replace(a,path=str(path)) if a.id==aid else a for a in self.refs]

    def bundle(self):
        return S.DataBundle("fixture",S.RevisionStamp(1,0,0,0),"s01",
            array_refs=tuple(self.refs),column_tables=tuple(self.tables),id_indices=tuple(self.indices),
            evidence=(S.Evidence("source",S.EvidenceKind.NATIVE_VIDEO,"test.mp4","test"),),
            clocks=(S.SourceClock(CLOCK_ID,1,12800,"source"),),frames=self.frames)


class ValueChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=DATA/"runs",prefix="ob_test_")
        self.addCleanup(self.temp.cleanup)
        self.f=CurrentFixture(self.temp.name)

    def codes(self,selection=None,bundle=None):
        return {i.code for i in validate_batches(None,bundle or self.f.bundle(),
            data_root=DATA,selections=(selection or self.f.selection,))}

    def test_valid_actual_values(self):
        self.assertEqual(self.codes(),set())

    def test_reversible_id_preserves_unicode_colons(self):
        value=observation_id("原:dataset",256256,"原/点:vertex/2")
        self.assertEqual(split_observation_id(value),("原:dataset",256256,"原/点:vertex/2"))

    def test_missing_foreign_key_is_rejected(self):
        self.f.replace_values("raw.frame_id",[b"missing-frame",b"O6j8L8TmfKo:256512"])
        self.assertIn("foreign_key_values",self.codes())

    def test_pts_frame_mismatch_is_rejected(self):
        self.f.replace_values("raw.native_pts",[256512,256256])
        self.assertIn("frame_pts_values",self.codes())

    def test_original_identity_cannot_silently_change(self):
        self.f.replace_values("raw.source_id",[b"changed",b"line/shared-id"])
        self.assertIn("original_identity_values",self.codes())

    def test_nonfinite_source_pixels_rejected(self):
        self.f.replace_values("raw.u",[np.nan,10])
        self.assertIn("raw_pixel_values",self.codes())

    def test_id_index_wrong_row_permutation_rejected(self):
        self.f.replace_values("raw.rows",[0,0])
        self.assertIn("id_index_row_permutation",self.codes())

    def test_small_selection_missing_stable_id_rejected(self):
        s=D.replace(self.f.selection,ids=("not-present",),selected_count=1)
        self.assertIn("selection_id_values",self.codes(s))

    def test_duplicate_small_id_selection_rejected(self):
        s=D.replace(self.f.selection,ids=(self.f.selection.ids[0],self.f.selection.ids[0]))
        self.assertIn("selection_duplicate_ids",self.codes(s))

    def test_stale_selection_generation_rejected(self):
        self.assertIn("selection_generation",self.codes(D.replace(self.f.selection,generation=2)))

    def test_mask_value_two_rejected(self):
        aid=self.f.save("mask",np.asarray([0,2],dtype="u1"),self.f.scalar,S.ArrayRole.SEMANTIC_FOREIGN_KEYS)
        s=S.SelectionRef("mask-selection","raw",1,S.SelectionMode.ROW_MASK,row_mask_array_id=aid,selected_count=1)
        self.assertIn("selection_mask_values",self.codes(s))

    def test_duplicate_row_selection_rejected(self):
        aid=self.f.save("rows",np.asarray([0,0],dtype="<u8"),self.f.scalar,S.ArrayRole.SEMANTIC_FOREIGN_KEYS)
        s=S.SelectionRef("row-selection","raw",1,S.SelectionMode.ROW_INDEX,row_index_array_id=aid,selected_count=2)
        self.assertIn("selection_duplicate_rows",self.codes(s))

    def test_out_of_range_row_selection_rejected(self):
        aid=self.f.save("rows",np.asarray([0,2],dtype="<u8"),self.f.scalar,S.ArrayRole.SEMANTIC_FOREIGN_KEYS)
        s=S.SelectionRef("row-selection","raw",1,S.SelectionMode.ROW_INDEX,row_index_array_id=aid,selected_count=2)
        self.assertIn("selection_row_values",self.codes(s))

    def test_missing_array_file_is_reported(self):
        bundle=self.f.bundle()
        refs=tuple(D.replace(a,path=str(self.f.root/"missing.npy")) if a.id=="raw.u" else a for a in bundle.array_refs)
        self.assertIn("current_array_read",self.codes(bundle=D.replace(bundle,array_refs=refs)))


class DeliveredS01(unittest.TestCase):
    def test_current_delivery_and_support_without_packet_reads(self):
        manifest=DATA/"runs/solve2_s01/observations/manifest.json"
        if not manifest.exists():
            self.skipTest("actual source delivery not available")
        original=Path.read_text
        def guarded(path,*args,**kwargs):
            if path.name in ("packet.json","source_model.json"):
                raise AssertionError("ordinary input must not open historical packet/model")
            return original(path,*args,**kwargs)
        with patch.object(Path,"read_text",guarded):
            delivery=load_import(manifest)
            self.assertEqual(delivery.report["s01_raw_rows"],355554)
            self.assertFalse(delivery.report["prepared_problem"])
            self.assertEqual(validate_batches(None,delivery.bundle,data_root=DATA,selections=delivery.selections),())
            self.assertEqual(validate_current_support(delivery,data_root=DATA),())
        a=S.raw_support(delivery.bundle,"s01.cross_a.358656")
        b=S.raw_support(delivery.bundle,"s01.cross_b.358656")
        self.assertEqual(len(a & b),14)  # Seven C midpoints, both raw edges each.
        self.assertEqual(len(a),20)
        self.assertEqual(len(b),19)  # Boundary-hit parent sample stayed excluded.
        self.assertTrue(all(x.local_variable_block_id is None for x in delivery.bundle.associations))


if __name__=="__main__":
    unittest.main()
