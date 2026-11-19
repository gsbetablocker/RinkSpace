"""Bind original material reads to the seven affected shots of the Ankara arena."""
from pathlib import Path
import shutil

import numpy as np

from board_profile_audit import MATERIALS, OUTPUT, PARENT
from review import camera_state
from shared import OUT, PREPARED, arrays, prepared_packet_path, read, save, sha, write
from side_repair import append
from source_bundle import native_run, relative


PRIOR = OUT/PARENT
OLD = PRIOR.parent
SHOTS = ('jin_s03','jin_s08','jin_s14','jin_s15','jin_s22','nayeon_s03','nayeon_s06')
FAR = OUTPUT/'far_prints/source_prints.npz'
NEAR = OUTPUT/'near_prints/jin_s14.npz'
RAIL = OUTPUT/'rail/jin_s14.npz'
EXTREMA = {'left':'xmin','right':'xmax','top':'ymin','bottom':'ymax'}


def build_inputs():
    directory = OUTPUT/'inputs'
    directory.mkdir(exist_ok=True)
    setup = read(PREPARED/'setup.json')
    baseline = PREPARED/'baseline.npz'
    shutil.copyfile(baseline, directory/'baseline.npz')
    state = camera_state(PARENT)
    report = {}
    for name in SHOTS:
        shot = setup['shots'][name]
        parent = prepared_packet_path(name)
        a = arrays(parent)
        before = a['fold'].copy()
        why = np.full(len(before),'',dtype='U55')
        why[(a['material']=='same_source_icon')&(before<9)] = 'reference_outline_from_patch_translation'
        additions = []
        if name == 'jin_s14':
            checked = arrays(OUTPUT/'rail/retire_verified_inputs.npz')
            assert checked['old_sha256'].item()==sha(parent)
            rr = checked['rows']
            for column in ('uv','pts','source_row','source_packet'):
                np.testing.assert_array_equal(a[column][rr], checked[column])
            np.testing.assert_array_equal(a['fold'][rr],checked['previous'])
            why[rr] = checked['reason']
            far = arrays(FAR)
            far['material'] = np.array(['ink_extent_'+EXTREMA[s] for s in far['direction']])
            far_typed = directory/'far_ink_axes.npz'
            save(far_typed,**far,original_packet_sha256=sha(FAR))
            near = arrays(NEAR)
            role = near['fold'].copy()
            exact = round(123.700*12800)
            role[(near['pts']==exact)&(role<9)] = 6
            # Exclude legacy reads on held print boundaries and from the
            # guarded full exposures; retain observations of unrelated material.
            near_families = np.unique(near['family'])
            close = np.isin(a['family'],near_families)
            period = (a['pts']>=near['pts'].min()) & (a['pts']<=near['pts'].max())
            prior = (before<5) & close & period
            held_paths = np.unique(near['path'][(near['fold']==6)&~near['station_fixed']])
            blocked_pts = np.unique(near['pts'][(near['fold']==5)|(near['fold']==9)])
            hold = prior & (np.isin(a['path'],held_paths)|np.isin(a['pts'],np.r_[blocked_pts,exact]))
            why[hold] = 'withheld_matching_material_or_exposure'
            # Ad26's faint inner ink remains visible on the last wide shot.
            # The printed outer circle is readable only on the early close view.
            why[(a['family']=='Ad26')&(a['pts']>=round(121.32*12800))&(before<9)]='outer_ink_not_resolved_on_source'
            # Guard-only ink landmarks have no independent curve identity in
            # this shot. Keep them in the source package, outside fit and check.
            far_roles = np.where(far['fold']==7,9,far['fold'])
            additions = [(RAIL, arrays(RAIL)['fold']), (far_typed,far_roles), (NEAR,role)]
        else:
            source = arrays(OUTPUT/'cross_rails'/(name+'.npz'))
            assert 'cross_column_deviation' in source and source['parent_packet_sha256'].item()==sha(parent)
            rr = source['row']
            np.testing.assert_array_equal(source['original_uv'],a['uv'][rr])
            np.testing.assert_array_equal(source['pts'],a['pts'][rr])
            np.testing.assert_array_equal(source['old_fold'],before[rr])
            bad = source['reason']!='same_material_edge'
            why[rr[bad]] = source['reason'][bad]
        retire = np.flatnonzero(why!='')
        if not len(retire):
            raise ValueError(('No material changes', name))
        a['fold'][retire] = 9
        for col in ('admission_reason','restored_admission_reason'):
            if col in a:
                a[col] = a[col].astype('U80')
                a[col][retire] = why[retire]
        revision = directory/'roles'/(name+'.npz')
        save(revision,row=retire,before=before[retire],after=a['fold'][retire],
             reason=why[retire],pts=a['pts'][retire],uv=a['uv'][retire],parent_sha256=sha(parent))
        shot['_pts'] = state[name+'_pts']
        new = [append(a,shot,p,role,91+j) for j,(p,role) in enumerate(additions)]
        del shot['_pts']
        dest = directory/(name+'.npz')
        save(dest,**a)
        shot['observations']=str(dest)
        report[name]=dict(parent=str(parent),parent_sha256=sha(parent),roles=str(revision),
                          retired=int(len(retire)),added=new,source_sha256=shot['source_sha256'])
    setup['parent']=str(PREPARED)
    write(directory/'source_audit.json',report)
    write(directory/'setup.json',setup)


def protect_paint():
    """Use the original paired grey-paint flanks and preserve distinct checks."""
    folder=OUTPUT/'material_inputs'
    folder.mkdir(exist_ok=True)
    name='jin_s14'
    setup=read(PREPARED/'setup.json')
    s=setup['shots'][name]
    packet=prepared_packet_path(name)
    a=arrays(packet)
    source=OLD/'appearance_audit/final_paint/jin_s14.npz'
    grey=arrays(source)
    assert np.allclose(grey['source_mid'],.5*(grey['f0']+grey['f1']),atol=1.e-4)
    # Each recovered pair refers back to a specific original source read.
    lookup={int(x):i for i,x in enumerate(a['origin_row'])}
    rows=np.array([lookup[int(o)] for o in grey['origin_row']])
    for k in ('pts','idx','path','uv'):
        np.testing.assert_array_equal(grey[k],a[k][rows])
    if np.any(a['station_fixed'][rows]):
        raise ValueError('A physical anchor cannot be replaced by a section midpoint')
    bad_shield=(a['source_packet']==19)&(a['family']=='Ad24')&(a['material']=='ink_extent_ymax')
    changed=np.r_[rows,np.flatnonzero(bad_shield)]
    before=a['fold'][changed].copy()
    a['fold'][changed]=9
    reason=np.where(bad_shield[changed],'red_fill_stops_above_white_shield_tip','paired_paint_supersedes_previous_read')
    for k in ('admission_reason','restored_admission_reason'):
        if k in a:
            a[k][changed]=reason
    save(folder/'replaced_s14.npz',row=changed,pts=a['pts'][changed],uv=a['uv'][changed],
         previous=before,reason=reason,parent_packet_sha256=sha(packet))
    target=folder/'paired_early_grey.npz'
    role=np.where((grey['role']=='repair_support')&~np.isin(grey['pts'],[1556992,1557248]),0,6).astype('i1')
    # 121.640/121.660 are kept out along with the original measurement holdouts.
    assert int(round(121.64*12800))==1556992 and int(round(121.66*12800))==1557248
    # Each read stays on the same finite path as its original. Initialize from
    # the previously localized source station; the midpoint gets a free station.
    station=arrays(PRIOR/'pixel_review/jin_s14.npz')['station'][rows]
    if not np.isfinite(station).all():raise ValueError('Missing source paint station')
    save(target,**{k:grey[k] for k in ('idx','pts','path','family','station_fixed','source_sha256')},
         station=station,uv=grey['source_mid'],material=np.repeat('direct_early_lane1_midpoint',len(role)),fold=role,
         source_file=str(source),source_file_sha256=sha(source),prior_role=grey['role'])
    s['_pts']=camera_state(PARENT)[name+'_pts']
    new=append(a,s,target,role,95)
    del s['_pts']
    out=folder/(name+'.npz')
    save(out,**a);s['observations']=str(out)
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    write(folder/'setup.json',setup)
    audit=read(PREPARED/'source_audit.json')
    audit[name].update(early_paint=new,early_paint_raw=str(source),retired_shield_vertical_held=int(bad_shield.sum()),
                       corrected_roles=str(folder/'replaced_s14.npz'),parent_packet=str(packet))
    write(folder/'source_audit.json',audit)


def pan_inputs():
    """Include the independently located paint on both sides of the fast pan."""
    folder=OUTPUT/'pan_inputs'
    folder.mkdir(exist_ok=True)
    setup=read(PREPARED/'setup.json');name='jin_s14';s=setup['shots'][name]
    packet=prepared_packet_path(name);a=arrays(packet);before=a['fold'].copy()
    red_file=OUTPUT/'pan_paint/reads.npz'
    red=arrays(red_file)
    assert np.array_equal(red['pts'],camera_state(PARENT)[name+'_pts'][red['idx']])
    fresh=red['family']=='goal_line_left'
    roles=np.where(fresh,red['fold'],9).astype('i1')
    reads=np.unique(red['pts'][fresh&(roles<9)])
    # All previous automatic points on this paint at the newly measured
    # exposures are superseded. Fixed material intersections keep their roles.
    remove=np.isin(a['pts'],reads)&(a['family']=='goal_line_left')&~a['station_fixed']&(before<9)
    a['fold'][remove]=9
    s['_pts']=camera_state(PARENT)[name+'_pts']
    # The native search used for the existing material sweep already required
    # grey annulus polarity or blue-filled crease ink at each source UV. Add
    # three earlier contact frames as development and keep the reported pan.
    ribbon=(a['material']=='surface_verified_boundary')&np.isin(a['family'],['goal_crease_left','F4'])
    anchors=np.rint(np.asarray([122.76,122.86,122.94])*12800).astype(int)
    add=ribbon&np.isin(a['pts'],anchors)&(before==6)
    for code in np.unique(a['source_packet'][add]):
        source=arrays(s['source_packets'][str(code)]['path'])
        rr=np.flatnonzero(add&(a['source_packet']==code));ix=a['source_row'][rr]
        np.testing.assert_array_equal(a['uv'][rr],source['uv'][ix])
        np.testing.assert_array_equal(a['path'][rr],source['keys'][source['path'][ix]])
        assert (source['contrast'][ix]>0).all()
    a['fold'][add]=0
    prior_active=ribbon&np.isin(a['pts'],np.rint(np.asarray([122.74,122.82,122.90])*12800).astype(int))&(before<5)
    typed=add|prior_active
    a['material']=a['material'].astype('U64')
    a['material'][typed]='pan_verified_annulus_or_crease'
    for k in ('admission_reason','restored_admission_reason'):
        if k in a:a[k][remove]='superseded_by_dual_red_paint_flanks'
    new=append(a,s,red_file,roles,96)
    a['material'][a['source_packet']==new['code']]='pan_double_red_paint'
    del s['_pts']
    out=folder/(name+'.npz')
    s['observations']=str(out);save(out,**a)
    rows=np.flatnonzero(remove|add|typed)
    changes=folder/'developed_ice.npz'
    save(changes,row=rows,pts=a['pts'][rows],uv=a['uv'][rows],previous=before[rows],current=a['fold'][rows],
         source_packet_sha256=sha(packet),physical_paint_rows=np.flatnonzero(typed))
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')
    write(folder/'setup.json',setup)
    audit=read(PREPARED/'source_audit.json')
    audit[name].update(pan_red=new,paired_red_source=str(red_file),pan_ice_role_changes=str(changes),
                       prior_automatic_red_removed=int(remove.sum()),newly_developed_arc_rows=int(add.sum()))
    write(folder/'source_audit.json',audit)


def qualify_circle_checks():
    """Retain only C0 midpoint checks whose physical paint was identified in the source."""
    import csv

    folder=OUTPUT/'circle_inputs'
    folder.mkdir(exist_ok=True)
    name='jin_s14';packet=prepared_packet_path(name)
    a=arrays(packet);setup=read(PREPARED/'setup.json')
    table=OUTPUT/'motion_check/circle_c0_source/held_rows.csv'
    with table.open(encoding='utf-8-sig') as file:
        entries=list(csv.DictReader(file))
    rows=np.array([int(e['packet_row_zero_based']) for e in entries])
    classification=np.array([e['classification'] for e in entries])
    np.testing.assert_array_equal(rows,np.flatnonzero((a['family']=='C0')&(a['fold']==6)&
                                                     (a['material']=='review_verified_stripe')))
    np.testing.assert_array_equal(a['pts'][rows],[int(e['source_pts']) for e in entries])
    np.testing.assert_array_equal(a['source_row'][rows],[int(e['source_stripe_row_zero_based']) for e in entries])
    np.testing.assert_array_equal(a['uv'][rows],[[float(e['source_u']),float(e['source_v'])] for e in entries])
    keep=classification=='C0_verified_pair'
    retired=rows[~keep]
    before=a['fold'].copy()
    a['fold'][retired]=9
    a['admission_reason']=a['admission_reason'].astype('U80')
    a['admission_reason'][retired]=np.char.add('unusable_as_C0_midpoint_',classification[~keep])
    np.testing.assert_array_equal(a['fold']<5,before<5)
    out=folder/(name+'.npz')
    save(out,**a)
    setup['shots'][name]['observations']=str(out)
    write(folder/'setup.json',setup)
    role_file=folder/'circle_roles.npz'
    save(role_file,row=retired,pts=a['pts'][retired],uv=a['uv'][retired],previous=before[retired],
         current=a['fold'][retired],classification=classification[~keep],source_packet_sha256=sha(packet))
    audit=read(PREPARED/'source_audit.json')
    audit[name].update(circle_parent=str(packet),circle_roles=str(role_file),circle_source_review=str(table),
                       old_C0_midpoints_confirmed=int(keep.sum()),unqualified_old_C0_midpoints=len(retired))
    write(folder/'source_audit.json',audit)
    shutil.copyfile(PREPARED/'baseline.npz',folder/'baseline.npz')


def solve_camera(name, label, *, fields=(), print_gain=1., flow_gain=None, parent=PARENT,
                 stride=2, ink_gain=1., rail_gain=1., early_grey=1800., cap_gain=1.,
                 red_gain=900., arc_gain=240.):
    folder=OUTPUT/label/name
    folder.mkdir(parents=True,exist_ok=True)
    if name!='jin_s14':
        raise ValueError('This calibration is bound to the Jin S14 source packets')
    prior=OLD/'physical_edges/jin_s14'
    weight=read(prior/'weights.json')
    # Same metric for the x and y coordinates of each read; one axis for
    # each ink extremum. Continuous rail cuts sample extended shared edges.
    for axis in EXTREMA.values():weight['ink_extent_'+axis]=95.*print_gain*ink_gain
    near=arrays(NEAR)
    for material in np.unique(near['material']):
        weight[str(material)]=(65. if material.endswith('corner') else 15.)*print_gain
    weight.update(front_face_top_junction=3., flat_navy_vertical_fascia_upper=3.,
                  kickplate_ice_contact=3.)
    for material in np.unique(arrays(RAIL)['material']):
        weight[str(material)]=(9.*cap_gain if 'cap' in material or 'lip' in material else 3.)*rail_gain
    weight['direct_early_lane1_midpoint']=early_grey
    weight['pan_double_red_paint']=red_gain
    weight['pan_verified_annulus_or_crease']=arc_gain
    flow=38 if flow_gain is None else flow_gain
    texture=OLD/'broad_paint/texture_jin_s14'
    write(folder/'weights.json',weight)
    options=['--mode','camera','--shots',name,'--stride',str(stride),'--unit-weights','--huber-delta','2.7',
        '--station-seed-dir',PRIOR/'pixel_review','--material-weights',folder/'weights.json',
        '--tracks','--dense-source-tracks','--source-visibility','--current-track-depths',
        '--texture-dir',texture,'--rebind-texture','--triplet-scale','4','--flow-gain',str(flow)]
    if fields:
        options+=['--rig','--rig-roll','--rig-kinematics','--rig-fields',*fields]
    native_run(folder,parent,options)


def local_material_motion(parent, label, centres, width, material_gains):
    """Refit only the motion supported by continuous observed material."""
    name = 'jin_s14'
    folder = OUTPUT/label/name
    state = camera_state(parent)
    setup = read(PREPARED/'setup.json')['shots'][name]
    seconds = state[name+'_pts']/12800.
    knot = seconds[0]+setup['first']+setup['dt']*np.arange(len(state[name+'_controls']))
    centre = np.asarray(centres)
    basis = np.maximum(0., 1.-((knot[:,None]-centre)/width)**2)**3
    indices = np.flatnonzero((seconds >= centre[0]-width)&(seconds <= centre[-1]+width))
    detail = folder/'detail'
    save(detail/(name+'.npz'),transient=indices,motion_basis=basis)
    weights = read(OUT/parent/'weights.json')
    weights.update(material_gains)
    write(folder/'weights.json',weights)
    options = ['--mode','camera','--shots',name,'--stride','2','--unit-weights',
               '--huber-delta','2.7','--station-seed-dir',PRIOR/'pixel_review',
               '--material-weights',folder/'weights.json','--tracks','--dense-source-tracks',
               '--source-visibility','--current-track-depths','--texture-dir',
               OLD/'broad_paint/texture_jin_s14','--rebind-texture','--triplet-scale','4',
               '--flow-gain','95','--detail-dir',detail,'--local-detail']
    native_run(folder,parent,options)


def refine_pan_motion(parent, label, red_gain):
    # The red stripe is identifiable over 18 consecutive exposures. Its
    # support on either side lets the value and derivatives taper to zero.
    local_material_motion(parent,label,[122.82,122.98,123.14],.245,{'pan_double_red_paint':red_gain})


def fit_board_and_paint(parent,label,line_gain,rail_gain,rig_fields):
    """Balance real paint midpoints and measured physical rails over the full shot."""
    folder=OUTPUT/label/'jin_s14'
    weights=read(OUT/parent/'weights.json')
    for material in ('direct_coloured_paint_flanks','original_two_sided_painted_line'):
        weights[material]=line_gain
    weights['measured_paired_paint_edges']=line_gain/3.
    weights['vertical_fascia_upper_junction']=rail_gain
    weights['kickplate_ice_contact']=rail_gain
    write(folder/'weights.json',weights)
    options=['--mode','camera','--shots','jin_s14','--stride','5','--unit-weights',
             '--huber-delta','2.7','--station-seed-dir',PRIOR/'pixel_review',
             '--material-weights',folder/'weights.json','--tracks','--dense-source-tracks',
             '--source-visibility','--current-track-depths','--texture-dir',
             OLD/'broad_paint/texture_jin_s14','--rebind-texture','--triplet-scale','4',
             '--flow-gain','95','--rig','--rig-roll','--rig-kinematics','--rig-fields',*rig_fields]
    native_run(folder,parent,options)


def combine(branches, name='reviewed'):
    b=camera_state(PARENT)
    for shot,branch in branches.items():
        c=camera_state(branch)
        np.testing.assert_array_equal(c['scene_values'],b['scene_values'])
        for suffix in ('_times','_pts'):
            np.testing.assert_array_equal(c[shot+suffix],b[shot+suffix])
        for suffix in ('_rigs','_controls','_active_folds'):
            b[shot+suffix]=c[shot+suffix]
    save(OUTPUT/name/'state.npz',**b)
    write(OUTPUT/name/'inputs.json',dict(parent=PARENT,branches=branches,
          prepared=str(PREPARED),common_field_sha256=str(b['scene_values_sha256'])))
