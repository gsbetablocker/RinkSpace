"""Explicit new course6 chart/maps only; no source graph installation or solve.

Original camera/world23 coordinates retain their indices. Six conditional
course coordinates precede the existing independent R/logf4 and statistical
bias3. A course initializer means the original geometry, not six zero values.
Actual geometry/plane installation is a separate mandatory caller operation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
import numpy as np
from scipy.sparse import csr_matrix
from installation_mount_candidate.mount_graph import MountOpticsChart
from shared_advertisement_pairs_candidate.paired_ad_graph import MountedPairedAdOpticsChart
from secondary_optics_candidate.secondary_optics_graph import PHYSICAL_COLUMN_SCALES
from z190_streaming_source_normal import SourceRowBlock

OLD_CAMERA_COUNT=34225
OLD_COMPLETE_COUNT=34232
CAMERA_COUNT=34231
PHYSICAL_COUNT=34235
COMPLETE_COUNT=34238
COURSE_INDICES=tuple(range(34225,34231))
REFERENCE_INDICES=tuple(range(34231,34235))
BIAS_INDICES=tuple(range(34235,34238))
PAIRED_INDICES=tuple(range(34213,34225))
OLD_WORLD_INDICES=tuple([0,1,2,3,4,*range(34189,34195),*PAIRED_INDICES])
WORLD_INDICES=OLD_WORLD_INDICES+COURSE_INDICES
OLD_TO_NEW=np.r_[np.arange(OLD_CAMERA_COUNT),np.arange(CAMERA_COUNT,COMPLETE_COUNT)]
OLD_TO_NEW.flags.writeable=False
HELD_INDICES=tuple(range(34126,34130))

COORDINATE_CONTRACT=dict(schema='rinkspace.explicit-course6-whole-source-coordinate-extension.v1',
 original_camera_parameter_count=OLD_CAMERA_COUNT,original_complete_parameter_count=OLD_COMPLETE_COUNT,
 current_camera_parameter_count=CAMERA_COUNT,current_physical_parameter_count=PHYSICAL_COUNT,
 current_complete_parameter_count=COMPLETE_COUNT,course_parameter_indices=list(COURSE_INDICES),
 independent_reference_rv_logf_indices=list(REFERENCE_INDICES),statistical_bias_indices=list(BIAS_INDICES),
 original_world23_indices=list(OLD_WORLD_INDICES),current_world29_indices=list(WORLD_INDICES),
 original_complete_to_current_parameter_indices=OLD_TO_NEW.tolist(),original_held4_indices=list(HELD_INDICES),
 original_shared_count=62,current_shared_count=68,private_reference_count=4,required_fresh_direction_count=72,
 original_camera_function_unaffected_by_course_coordinates=True,source_geometry_can_change_with_course=True,
 course_zero_delta_means_explicit_original_course_initial_values_not_zero_vector=True,
 new_physical_world_freedoms=6,new_camera_or_statistical_freedoms=0,
 new_source_rows_or_priors=0,source_pixels_roles_sigmas_or_held_changed=False,
 geometry_and_finite_planes_must_be_installed_before_original_profile_update=True,
 new_geometry_is_not_hardware_measurement=True,source_graph_installed=False,accuracy_accepted=False)
COORDINATE_CONTRACT_SHA256=hashlib.sha256(json.dumps(COORDINATE_CONTRACT,sort_keys=True,
 separators=(',',':')).encode('utf8')).hexdigest()

def require(value,message):
 if not value:raise ValueError(message)

def validate_complete_q(q):
 value=np.asarray(q,float)
 require(value.shape==(COMPLETE_COUNT,)and np.isfinite(value).all(),'Explicit finite full34238 course source state required')
 return value

def old_complete_q(q):
 """Exact original full34232 coordinate order, including Rf4 and bias3."""
 return validate_complete_q(q)[OLD_TO_NEW]

def extend_complete_state(old_q,course_initial):
 old=np.asarray(old_q,float);course=np.asarray(course_initial,float)
 require(old.shape==(OLD_COMPLETE_COUNT,)and course.shape==(6,)and
  np.isfinite(old).all()and np.isfinite(course).all(),'Original full34232 plus declared course6 initial values required')
 result=np.r_[old[:OLD_CAMERA_COUNT],course,old[OLD_CAMERA_COUNT:]]
 require(np.array_equal(result[OLD_TO_NEW],old),'Original complete state was changed during extension')
 return result

def map_old_jacobian(matrix):
 """Monotone column remap, preserving duplicates, explicit zeros and storage."""
 require(getattr(matrix,'format',None)=='csr'and matrix.shape[1]==OLD_COMPLETE_COUNT,
  'Original complete34232 CSR storage required, no implicit CSC canonicalization')
 require(not len(matrix.indices)or (matrix.indices.min()>=0 and matrix.indices.max()<OLD_COMPLETE_COUNT),
  'Original CSR contains a non-original source coordinate')
 indices=OLD_TO_NEW[matrix.indices].astype(matrix.indices.dtype,copy=False)
 result=csr_matrix((matrix.data,indices,matrix.indptr),shape=(matrix.shape[0],COMPLETE_COUNT),copy=False)
 result.has_sorted_indices=matrix.has_sorted_indices;result.has_canonical_format=matrix.has_canonical_format
 require(np.array_equal(result.data,matrix.data)and np.array_equal(result.indptr,matrix.indptr),
  'Original row pointer/data storage changed')
 return result

def extend_shared_indices(old_shared):
 source=np.asarray(old_shared,int)
 require(source.shape==(62,)and len(set(source.tolist()))==62 and
  source.min()>=0 and source.max()<OLD_COMPLETE_COUNT and
  set(range(34229,34232)).issubset(source)and not set(range(34225,34229)).intersection(source),
  'Original62 shared slots with bias3 and without independent referenceRf4 required')
 result=np.r_[OLD_TO_NEW[source],COURSE_INDICES]
 require(len(set(result.tolist()))==68 and not set(REFERENCE_INDICES).intersection(result),
  'Exactly68 shared and separate4 reference directions required')
 return result


def append_course_chart_optics_csr(matrix,optics,indices,original_parameter_count=34204):
 """Insert lens3 before pair/course world columns, preserving old storage.

 Input must already be sorted CSR. Duplicates and explicit zeros remain in
 their original within-row order, and canonical flags are retained. Mount
 slots are inserted later by the original insert_mount_csr helper.
 """
 require(getattr(matrix,'format',None)=='csr','Explicit original CSR required; no format conversion')
 m=matrix;v=np.asarray(optics,float);slots=np.asarray(indices,dtype=m.indices.dtype)
 require(original_parameter_count==34204 and m.shape[1]==CAMERA_COUNT and slots.shape==(3,)and
  np.array_equal(slots,np.arange(slots[0],slots[0]+3))and slots[0]in(34204,34207,34210)and
  not np.any(m.indices<0)and not np.any((m.indices>=34195)&(m.indices<PAIRED_INDICES[0]))and
  not np.any(m.indices>=CAMERA_COUNT)and m.has_sorted_indices and
  v.ndim==2 and v.shape[1]==3 and len(v)<=m.shape[0]and np.isfinite(v).all(),
  'Sorted original CSR plus explicit pair12/course6, lens3 and whole34231 required')
 active=v!=0.;counts=np.zeros(m.shape[0],dtype=m.indptr.dtype);counts[:len(v)]=active.sum(axis=1)
 pointer=m.indptr+np.r_[0,np.cumsum(counts,dtype=m.indptr.dtype)]
 data=np.empty(int(pointer[-1]),dtype=m.data.dtype);columns=np.empty(len(data),dtype=m.indices.dtype)
 for lo in range(0,m.shape[0],65536):
  hi=min(lo+65536,m.shape[0]);left,right=map(int,m.indptr[[lo,hi]])
  sizes=np.diff(m.indptr[lo:hi+1]);rows=np.repeat(np.arange(lo,hi,dtype=m.indptr.dtype),sizes)
  inner=np.arange(right-left,dtype=m.indptr.dtype)-np.repeat(m.indptr[lo:hi]-left,sizes)
  old_columns=m.indices[left:right];tail=old_columns>=PAIRED_INDICES[0]
  dest=pointer[rows]+inner+counts[rows]*tail
  data[dest],columns[dest]=m.data[left:right],old_columns
  stop=min(hi,len(v))
  if stop>lo:
   tails=np.bincount(rows[tail]-lo,minlength=hi-lo)
   rr,cc=np.nonzero(active[lo:stop]);rank=np.cumsum(active[lo:stop],axis=1)-1
   old_prefix=sizes[:stop-lo]-tails[:stop-lo]
   dest=pointer[lo+rr]+old_prefix[rr]+rank[rr,cc]
   data[dest],columns[dest]=v[lo+rr,cc],slots[cc]
 result=csr_matrix((data,columns,pointer),shape=m.shape,copy=False)
 result.has_sorted_indices=m.has_sorted_indices;result.has_canonical_format=m.has_canonical_format
 return result


@dataclass(frozen=True)
class MountedCourseOpticsChart(MountedPairedAdOpticsChart):
 full_parameter_count:int=CAMERA_COUNT
 course_parameter_indices:tuple[int,...]=COURSE_INDICES
 coordinate_contract_sha256:str=COORDINATE_CONTRACT_SHA256

 def delta(self,q):
  value=np.asarray(q,float)
  require(value.shape==(CAMERA_COUNT,)and np.isfinite(value).all()and
   self.full_parameter_count==CAMERA_COUNT and self.original_parameter_count==34204 and
   self.paired_ad_parameter_indices==PAIRED_INDICES and self.course_parameter_indices==COURSE_INDICES and
   self.coordinate_contract_sha256==COORDINATE_CONTRACT_SHA256 and len(self.indices)==3 and
   tuple(self.indices)==tuple(range(self.indices[0],self.indices[0]+3))and self.indices[0]in(34204,34207,34210)and
   tuple(self.mount_indices)==tuple(range(34195+self.indices[0]-34204,34198+self.indices[0]-34204)),
   'Authenticated whole34231 paired/mount/lens/course6 camera chart required; no tail truncation')
  return value[np.asarray(self.indices)]*np.asarray(PHYSICAL_COLUMN_SCALES)

 @classmethod
 def from_paired(cls,original):
  require(type(original)is MountedPairedAdOpticsChart and original.full_parameter_count==OLD_CAMERA_COUNT,
   'Explicit original paired34225 optics chart required')
  return cls(original.installation_id,original.indices,original.original_parameter_count,
   original.physical_steps,original.mount_indices)

 def manifest(self,base):
  # Skip the parent's hardcoded34225 wholeq metadata; the actual inherited
  # calibration method still applies exactly the same lens then mount order.
  result=MountOpticsChart.manifest(self,base)
  result.update(whole_physical_parameter_count=CAMERA_COUNT,
   current_camera_chart='explicit_mount9_optics9_pair12_course6_whole_camera_state',
   mount_parameter_indices=list(self.mount_indices),paired_ad_parameter_indices=list(PAIRED_INDICES),
   course_parameter_indices=list(COURSE_INDICES),world_parameter_indices=list(WORLD_INDICES),
   original_pair_camera_parameter_count=OLD_CAMERA_COUNT,
   original_camera_function_unaffected_by_course_coordinates=True,
   coordinate_contract_sha256=COORDINATE_CONTRACT_SHA256)
  return result


class CourseBeforeMappedProfileUpdate:
 """Mandatory caller setter installs the current world/planes before weighting."""
 def __init__(self,original_update,set_course_state):
  require(callable(original_update)and callable(set_course_state),'Real original updater and explicit course installer callback required')
  self.original_update=original_update;self.set_course_state=set_course_state
 def __call__(self,q):
  value=validate_complete_q(q)
  self.set_course_state(value)
  return self.original_update(old_complete_q(value))
 def checkpoint_arrays(self):return self.original_update.checkpoint_arrays()
 def iter_checkpoint_arrays(self):return self.original_update.iter_checkpoint_arrays()
 def restore_checkpoint(self,data):return self.original_update.restore_checkpoint(data)


class MappedPP6WorkingPriorBlock:
 """The same engineering six rows on full34238; course derivatives are zero."""
 def __init__(self,original):
  require(original.rows==6 and original.mode=='principal_point_only6'and
   original.contract['parameter_count']==OLD_COMPLETE_COUNT and original.contract['radial_rows_admitted']is False,
   'Only the source-qualified original PP6 prior, no radial seed guess, required')
  self.original=original;self.rows=6;self.mode=original.mode
  self.quadratic=original.quadratic;self.scales=original.scales
  self.contract=dict(schema='rinkspace.course-chart-mapped-optics-PP6-working-prior.v1',
   original_working_prior_contract=original.contract,coordinate_contract_sha256=COORDINATE_CONTRACT_SHA256,
   parameter_count=COMPLETE_COUNT,original_complete_parameter_count=OLD_COMPLETE_COUNT,
   camera_parameter_count=CAMERA_COUNT,physical_parameter_count=PHYSICAL_COUNT,
   original_complete_to_current_parameter_indices=OLD_TO_NEW.tolist(),rows=6,radial_rows=0,
   actual_current_native_PP_target_px=[960.,540.],engineering_sigma_native_px=32.,
   course_direct_prior_derivatives_zero=True,original_prior_objective_values_preserved_under_coordinate_map=True,
   additional_prior_rows=0,new_hard_bounds=0,manufacturer_measurement_claim=False,accuracy_accepted=False)
 def evaluate(self,q,jacobian=True):
  original=self.original.evaluate(old_complete_q(q),jacobian)
  if not jacobian:return original
  residual,matrix=original
  return residual,map_old_jacobian(matrix)
 def current_calibrations(self,q):return self.original.current_calibrations(old_complete_q(q))
 def review(self,q):
  old=self.original.review(old_complete_q(q))
  return dict(old,contract=self.contract,original_working_prior_contract=self.original.contract)
 def source_block(self,q,jacobian=True,*,original_start=4624626):
  require(isinstance(original_start,int)and original_start>=0,'Explicit preceding source scalar row count required')
  original=self.evaluate(q,jacobian)
  residual,matrix=original if jacobian else(original,None)
  ledger=dict(family='explicit_optics_engineering_working_prior',original_start=original_start,
   original_stop=original_start+6,source_role='engineering_working_hypothesis_prior',
   optical_prior_contract=self.contract)
  return SourceRowBlock(residual,matrix,self.quadratic,self.scales,ledger)
