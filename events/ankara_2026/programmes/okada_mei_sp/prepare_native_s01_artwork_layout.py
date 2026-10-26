"""Keep each complete printed design together, instead of fitting separate lines.

This is an explicit model revision. Reference artwork is not declared measured;
template/source errors require source repair, never independent letter/line warp.
Whole installation/size and existing analytic panel relationships remain fitted.
"""
from pathlib import Path
import argparse
import io
import shutil
import numpy as np

import prepare_native_s01 as prep
import prepare_native_s01_print_layout as layout
import prepare_native_s01_panel_domain as panel
from rinkspace.domain.scene_assets import read_scene_assets, scene_content_identity, snapshot_scene
from rinkspace.domain.parameters import SceneParameterModel
from rinkspace.domain.print_parameters import parameterize_print_layout
from rinkspace.path_policy import guard_project_write


OUT = prep.RUN/'print_artwork_layout'
DEFINITION = prep.SCENE.parent.parent/'print_layout_components.json'
PLAN = prep.RUN/'print_component_layout/whole_artwork_definition.json'


def stage():
    if OUT.exists() or PLAN.exists():
        raise ValueError('Do not overwrite an existing staged artwork revision')
    plan = prep.read(DEFINITION)
    records = {r['template_id']:r for r in plan['records']}
    for ref, assets in read_scene_assets(prep.read(prep.SCENE), prep.SCENE):
        record = records[ref['template_id']]
        groups = record['groups']
        members = {path for group in groups for path in group['path_ids']}
        template = next(t for t in assets['templates'] if t['id']==ref['template_id'])
        if len(groups)>1:
            evidence = []
            for group in groups:
                source = Path(group['evidence']['source_image'])
                if prep.digest(source) != group['evidence']['source_image_sha256']:
                    raise ValueError('Reviewed source image changed: '+str(source))
                evidence.append(group['evidence'])
            record['groups'] = [dict(
                id=f"Ad{record['advertisement']}/complete printed artwork",
                path_ids=[p['id'] for p in template['paths'] if p['id'] in members],
                fixed_in_template_chart=False,
                evidence=dict(kind='single_printed_design_with_fixed_relative_typography_and_logo',
                    components=[g['id'] for g in groups], component_source_reviews=evidence,
                    decision='All ordinary contours, lines of text, logos and holes share one map. Independent inter-line stretching and overlap are not physical print freedoms.',
                    reference_artwork_status='existing source-derived reference, not a surveyed printing master',
                    physical_size_measured=False, source_paths_and_UV_preserved=True))]
        record['source_asset_sha256'] = ref['sha256']
    plan.update(reason='Far camera initialisation reproduced Ad12 cross-line overlap despite the per-word model; preserve complete printed designs',
        numerical_policy='One positive affine map for each complete ordinary printed artwork; analytic frames retain their existing construction; no per-line or per-logo deformation',
        reference_artwork_is_not_surveyed=True,
        remaining_unknowns='Whole dimensions/installation and the existing relationship to an explicit analytic panel; camera freedoms unchanged',
        source_point_changes=False)
    prep.write(PLAN, plan)
    layout.stage(OUT, PLAN, preserve_unchanged_shapes=True)
    shutil.copyfile(DEFINITION, OUT/'before_layout_definition.json')
    report=prep.read(OUT/'migration.json')
    report.update(installed=False,definition_path=str(DEFINITION),definition_before_sha256=prep.digest(DEFINITION),
                  definition_after_sha256=prep.digest(PLAN))
    prep.write(OUT/'migration.json',report)


def install():
    report=prep.read(OUT/'migration.json')
    if prep.digest(DEFINITION)!=report['definition_before_sha256'] or prep.digest(PLAN)!=report['definition_after_sha256']:
        raise ValueError('Source layout definition changed during review')
    # The generic panel installer checks the frozen model, input, accepted state,
    # complete native validation and all-template topology before any mutation.
    panel.OUT=OUT
    panel.install()
    pending=guard_project_write(DEFINITION.with_name(DEFINITION.name+'.artwork_pending'))
    pending.write_bytes(PLAN.read_bytes())
    pending.replace(guard_project_write(DEFINITION))
    report=prep.read(OUT/'migration.json')
    report.update(production_installed=True,installed_definition_sha256=prep.digest(DEFINITION))
    prep.write(OUT/'migration.json',report)


def anchor_ink_rectangle():
    """Ad5's rectangle is its blue/green ink graphic, not a separate panel."""
    if not (OUT/'stop_requested').exists():
        raise ValueError('Stop the staged numeric run before revising its model')
    backup=OUT/'before_ink_anchor'
    if backup.exists():
        raise ValueError('Existing artwork subrevision must not be overwritten')
    scene_path=OUT/'scene/scene.json';scene=prep.read(scene_path)
    migration=prep.read(OUT/'migration.json')
    if scene_content_identity(scene,scene_path)!=migration['after_scene_revision']:
        raise ValueError('Staged scene changed')
    old_model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    with np.load(io.BytesIO((OUT/'accepted_pixels_latest.npz').read_bytes())) as archive:
        saved={key:archive[key] for key in archive.files}
    backup.mkdir()
    snapshot_scene(scene_path,backup/'scene')
    for name in ('accepted_pixels_latest.npz','migration.json','fit_input_validation.json','activity_pixels.json'):
        shutil.copyfile(OUT/name,backup/name)
    shutil.copyfile(PLAN,backup/'layout_definition.json')
    plan=prep.read(PLAN);record=next(r for r in plan['records'] if r['advertisement']==5)
    if len(record['groups'])!=1:
        raise ValueError('Expected complete ordinary artwork group')
    group=record['groups'][0]
    group['fixed_in_template_chart']=True
    group['evidence']['analytic_reference_is_same_printed_artwork']=dict(
        path_ids=['Ad5.bottom_edge/path0','Ad5.left_vertical/path0','Ad5.right_vertical/path0','Ad5.top_edge/path0'],
        meaning='Blue/green MEIWA rectangular graphic edges, not the white advertisement panel',
        source_image=str(prep.RUN/'print_shape_audit/Ad05_source_417024.png'),source_pts=417024,
        decision='The text, anniversary emblem, colored bars and their analytic rectangle share the fixed reference layout. Whole size and placement remain fitted.')
    plan['remaining_unknowns']='Whole dimensions/installation; complete ink layout relative to the five explicit physical panels only; camera freedoms unchanged'
    prep.write(PLAN,plan)
    reference,assets=next((r,a) for r,a in read_scene_assets(scene,scene_path) if r['advertisement_number']==5)
    updated=parameterize_print_layout(assets,{reference['template_id']:record['groups']})
    target=scene_path.parent/reference['path'];prep.write(target,updated);reference['sha256']=prep.digest(target)
    scene['print_layout_revision']['definition_sha256']=prep.digest(PLAN)
    scene['print_layout_revision']['analytic_ink_anchor']='Ad5 graphic frame is part of the same printed artwork'
    prep.write(scene_path,scene)
    new_model=SceneParameterModel(scene,scene_path,optimize_scene=True)
    before=saved['scene_values'].copy();values=before.copy();reset=[]
    for identity,b in new_model.initial_state.blocks.items():
        if not b['available']:continue
        old=old_model.initial_state.blocks[identity]
        if (old['offset'],old['size'])!=(b['offset'],b['size']):
            raise ValueError('Unexpected storage change in ink-frame relation')
        if old['independent_local_indices']!=b['independent_local_indices']:
            if b['kind']!='print_shape' or b['independent_local_indices']:
                raise ValueError('Unexpected freedom change')
            values[b['offset']:b['offset']+b['size']]=new_model.initial_state[identity]
            reset.append(dict(block=identity,before=before[b['offset']:b['offset']+b['size']].tolist(),after=new_model.initial_state[identity].tolist()))
    if len(reset)!=1:raise ValueError('Expected only the Ad5 ink-frame map to be removed')
    saved['scene_values']=values;np.savez(OUT/'seed_frame_anchor.npz',**saved)
    mapping=prep.read(OUT/'mapping.json');mapping['scene_revision']=scene_content_identity(scene,scene_path);prep.write(OUT/'mapping.json',mapping)
    migration.update(after_scene_revision=mapping['scene_revision'],definition_after_sha256=prep.digest(PLAN),
        ink_frame_revision=dict(ad=5,reset=reset,before=str(backup),camera_and_other_current_coordinates_preserved=True),
        active_coordinates={**migration['active_coordinates'],'print_shape':20})
    next(a for a in migration['assets'] if a['ad']==5)['after_sha256']=reference['sha256']
    prep.write(OUT/'migration.json',migration)
    print('Ad5 complete ink layout anchored; remaining internal coordinates: 20')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['stage','anchor-ink-rectangle','install'])
    args=parser.parse_args()
    if args.action=='stage':stage()
    elif args.action=='anchor-ink-rectangle':anchor_ink_rectangle()
    else:install()
