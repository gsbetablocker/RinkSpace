"""Review authored knee/elbow coordinates on the saved shaped SMPL-X skeleton."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.spatial.transform import Rotation

from calibration.program_tracking.body_tracking.anatomical_chart import JointChart
from calibration.program_tracking.body_tracking.footmr_contract import fk_rotation
from calibration.program_tracking.body_tracking.programme_joint_limits import JOINTS, NAMES, authored_bases
from calibration.program_tracking.body_tracking.rebuild_prepare import digest


CASES = ("Neutral", "Positive flexion", "Mild extension", "Deep flexion", "Off-axis")
TWIST_DEG = np.array([0., 45., -8., 130., 35.])
SWING_DEG = np.array([[0., 0.], [0., 0.], [0., 0.], [0., 0.], [15., 20.]])
CHAINS = ((1, 4, 7), (2, 5, 8), (16, 18, 20), (17, 19, 21))
RIGHT_KNEE_PTS = 2081280


def bone_fk(rest, parents, local_R):
    """Return root-local frames and centres from the archived rest offsets."""
    local = np.broadcast_to(np.eye(3), (len(local_R), 22, 3, 3)).copy()
    local[:, 1:] = local_R
    frames = fk_rotation(local, parents)
    centres = np.broadcast_to(rest[:22], (len(local), 22, 3)).copy()
    for joint in range(1, 22):
        parent = parents[joint]
        centres[:, joint] = centres[:, parent] + frames[:, parent] @ (rest[joint] - rest[parent])
    return centres, frames


def synthesize(rest, parents, chart):
    joint_ids = JOINTS[:4]
    twist = np.deg2rad(np.column_stack((TWIST_DEG, np.zeros((len(CASES), 2)))))
    swing = np.deg2rad(np.column_stack((np.zeros(len(CASES)), SWING_DEG)))
    basis = chart.basis[:4]
    neutral = chart.rest[:4]
    local = np.broadcast_to(np.eye(3), (len(CASES), 21, 3, 3)).copy()
    local[:, joint_ids - 1] = (neutral @ basis @ Rotation.from_rotvec(swing).as_matrix()[:, None]
                                @ Rotation.from_rotvec(twist).as_matrix()[:, None]
                                @ basis.swapaxes(-1, -2))
    centres, _ = bone_fk(rest, parents, local)
    rotations = local[:, joint_ids - 1]
    angles = JointChart(neutral, basis, np.zeros((len(CASES), len(joint_ids)))).evaluate(rotations)
    np.testing.assert_allclose(np.degrees(angles.twist), np.broadcast_to(TWIST_DEG[:, None], (5, 4)),
                               rtol=0, atol=1e-11)
    np.testing.assert_allclose(np.degrees(angles.swing_vector), np.broadcast_to(SWING_DEG[:, None], (5, 4, 2)),
                               rtol=0, atol=1e-11)
    assert not (angles.singular | angles.outside_branch).any()
    np.testing.assert_allclose(centres[0], rest[:22], rtol=0, atol=1e-13)

    anterior = np.cross(rest[1] - rest[2], rest[12] - rest[0])
    anterior /= np.linalg.norm(anterior)
    records, plot_chains = [], []
    for i, (parent, joint, child) in enumerate(CHAINS):
        np.testing.assert_array_equal(parents[[joint, child]], [parent, joint])
        chain = centres[:, [parent, joint, child]]
        distal = chain[:, 2] - chain[:, 1]
        coords = distal @ basis[i]
        planar = np.degrees(np.arctan2(coords[:, 2], coords[:, 1]))
        np.testing.assert_allclose(planar[:4], TWIST_DEG[:4], rtol=0, atol=1e-11)
        signed_shift = float((distal[1] - distal[0]) @ anterior)
        assert signed_shift < 0 if i < 2 else signed_shift > 0
        lengths = np.linalg.norm(np.diff(chain, axis=1), axis=2)
        np.testing.assert_allclose(lengths, np.broadcast_to(lengths[0], lengths.shape), rtol=0, atol=1e-13)
        plot_chains.append((chain - chain[:, 1, None]) @ basis[i] / lengths[0, 1])
        records.append(dict(joint_name=NAMES[i], native_joint_ids=[parent, joint, child],
                            anatomical_basis=basis[i].tolist(),
                            positive_flexion_distal_shift_dot_anterior_rest_units=signed_shift,
                            cases=[dict(name=name, local_R=rotations[k, i].tolist(),
                                        twist_deg=float(np.degrees(angles.twist[k, i])),
                                        swing_vector_deg=np.degrees(angles.swing_vector[k, i]).tolist(),
                                        swing_deg=float(np.degrees(angles.swing[k, i])),
                                        projected_distal_bend_deg=float(planar[k]),
                                        centres_shaped_root_local=chain[k].tolist())
                                   for k, name in enumerate(CASES)]))
    return records, plot_chains


def actual_knee(fk, archive, rest, parents, chart):
    pts = fk["source_pts"]
    selected = np.flatnonzero(pts == RIGHT_KNEE_PTS)
    if len(selected) != 1:
        raise ValueError("Expected one exposure at source PTS 2081280")
    i = int(selected[0])
    column = NAMES.index("right_knee")
    local = fk["local_R"][i:i + 1]
    centres, frames = bone_fk(rest, parents, local)
    np.testing.assert_allclose(frames[0], fk["frames"][i], rtol=0, atol=2e-9)
    replay = (centres[0] - rest[0]) * float(fk["body_scale"]) @ fk["Q"][i].T + fk["P"][i]
    saved = fk["native_world"][i]
    np.testing.assert_allclose(replay, saved, rtol=0, atol=2e-9)
    rotation = local[:, JOINTS[column] - 1]
    angles = chart.evaluate(rotation, sample_indices=[i], joint_indices=[column])
    assert not (angles.singular | angles.outside_branch).any()
    np.testing.assert_allclose(angles.twist[0], archive["twist"][i, column], rtol=0, atol=2e-9)
    np.testing.assert_allclose(angles.swing_vector[0], archive["swing_vector"][i, column], rtol=0, atol=2e-9)
    chain = saved[list(CHAINS[column])]
    proximal, distal = chain[[0, 2]] - chain[1]
    internal = np.degrees(np.arctan2(np.linalg.norm(np.cross(proximal, distal)), proximal @ distal))
    twist_deg = float(np.degrees(angles.twist[0]))
    principal = (twist_deg + 180.) % 360. - 180.
    neighbours = slice(i - 2, i + 3)
    return dict(source_pts=int(pts[i]), source_seconds=float(fk["source_seconds"][i]),
                source_time_base=[1, 12800], exposure_index_zero_based=i, native_joint_id=int(JOINTS[column]),
                local_R=rotation[0].tolist(), rest_R=chart.rest[column].tolist(),
                anatomical_basis=chart.basis[column].tolist(),
                reference_twist_anchor_deg=float(np.degrees(chart.reference_twist[i, column])),
                fixed_branch_twist_deg=twist_deg, same_chart_principal_twist_deg=principal,
                winding_turns=round((twist_deg - principal) / 360.),
                off_axis_swing_deg=float(np.degrees(angles.swing[0])),
                swing_vector_deg=np.degrees(angles.swing_vector[0]).tolist(),
                centre_internal_angle_deg=float(internal), centre_angle_supplement_deg=float(180. - internal),
                centre_joint_ids=list(CHAINS[column]), centres_world_solver_units=chain.tolist(),
                fk_replay_max_abs_solver_units=float(np.max(np.abs(replay - saved))),
                adjacent_source_pts=pts[neighbours].tolist(),
                adjacent_fixed_branch_twist_deg=np.degrees(archive["twist"][neighbours, column]).tolist())


def plot_chains(chains, output):
    blue, orange, grey = "#16697a", "#b74d24", "#4b5d70"
    fig, axes = plt.subplots(len(CASES), len(chains), figsize=(12, 10.5))
    fig.subplots_adjust(left=.125, right=.985, top=.865, bottom=.118, hspace=.16, wspace=.10)
    fig.suptitle("Knee and elbow axes on the shaped rest skeleton", x=.125, y=.975,
                 ha="left", fontsize=20, weight="bold", color="#152736")
    fig.text(.125, .928, "Authored SMPL-X coordinates for this model. Angles are synthetic test inputs.",
             fontsize=10, color=grey)
    for col, (name, chain) in enumerate(zip(NAMES[:4], chains)):
        axes[0, col].set_title(name.replace("_", " ").title(), fontsize=12, weight="bold", color="#152736")
        for row, ax in enumerate(axes[:, col]):
            ax.set_xlim(-1.30, 1.30)
            ax.set_ylim(-1.12, 1.30)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.axhline(0, color="#e1e7ec", lw=.75, zorder=0)
            ax.axvline(0, color="#e1e7ec", lw=.75, zorder=0)
            ax.plot(chain[0, 1:, 2], -chain[0, 1:, 1], "--", lw=2.7, color="#bcc8d1")
            points = chain[row]
            ax.plot(points[:2, 2], -points[:2, 1], "o-", color=grey, lw=2.7, ms=4.2, zorder=3)
            color = orange if row == len(CASES) - 1 else blue
            ax.plot(points[1:, 2], -points[1:, 1], "o-", color=color, lw=3, ms=4.2, zorder=2)
            if col == 0:
                label = f"{CASES[row]}\n{TWIST_DEG[row]:+g}° twist"
                if row == len(CASES) - 1:
                    label += "\n25° swing"
                ax.text(-.63, .48, label, transform=ax.transAxes, va="center",
                        ha="left", fontsize=9.3, linespacing=1.45, color="#152736")
            if row == len(CASES) - 1:
                ax.text(.64, .97, "Side view", transform=ax.transAxes, color=grey, fontsize=7.3, ha="center")
                inset = ax.inset_axes([.48, .53, .32, .40])
                inset.set(xlim=(-.54, .54), ylim=(-1.09, .14), aspect="equal")
                inset.axis("off")
                inset.plot(chain[0, 1:, 0], -chain[0, 1:, 1], "--", color="#a6b7c5", lw=1.4)
                inset.plot(points[1:, 0], -points[1:, 1], "o-", color=orange, lw=1.8, ms=2.5)
    handles = [Line2D([], [], color=grey, lw=2.7, label="Proximal bone"),
               Line2D([], [], color=blue, lw=2.7, label="Posed distal bone"),
               Line2D([], [], color="#a6b7c5", lw=2.2, ls="--", label="Neutral distal bone"),
               Line2D([], [], color=orange, lw=2.7, label="Combined flexion and swing")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.115, .058), ncol=4, frameon=False,
               handlelength=2.4, fontsize=9.5)
    fig.text(.125, .038, "Equal scale within each panel. Right is the authored bend direction. Side view shows the out-of-plane offset.",
             fontsize=9, color=grey)
    fig.text(.125, .017, "Neutral is the shaped model rest pose. The illustrated angles do not define a clinical zero or a range of motion.",
             fontsize=9, color=grey)
    fig.savefig(output, dpi=170, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True, help="Directory containing the two baseline NPZ archives")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anatomical_path = args.baseline.resolve() / "baseline_anatomical_frames.npz"
    fk_path = args.baseline.resolve() / "baseline_fk.npz"
    with np.load(anatomical_path, allow_pickle=False) as a, np.load(fk_path, allow_pickle=False) as f:
        np.testing.assert_array_equal(a["source_pts"], f["source_pts"])
        np.testing.assert_array_equal(a["joint_names"], NAMES)
        np.testing.assert_array_equal(a["joint_ids"], JOINTS)
        for a_key, f_key in (("rest_joint_centres", "rest_J"), ("parents", "parents"),
                             ("body_scale", "body_scale"), ("length_ratio", "length_ratio"),
                             ("local_R", "local_R"), ("FK_frames", "frames"), ("global_root_R", "Q")):
            np.testing.assert_allclose(a[a_key], f[f_key], rtol=0, atol=2e-9)
        rest, parents = a["rest_joint_centres"], a["parents"]
        basis = authored_bases(rest)
        np.testing.assert_allclose(a["anatomical_basis"], basis, rtol=0, atol=2e-12)
        chart = JointChart(a["rest_R"], basis, a["reference_twist"])
        synthetic, chains = synthesize(rest, parents, chart)
        actual = actual_knee(f, a, rest, parents, chart)

    args.output.mkdir(parents=True, exist_ok=False)
    plot_path = args.output / "joint_axes.png"
    plot_chains(chains, plot_path)
    report = dict(schema="rinkspace.joint-axis-review.v1",
                  interpretation="作者化 SMPL-X 中立坐标。合成角仅用于验证左右正号、离轴分解与 FK。"
                                 "模型零位不等于临床零位，图示角度不用于确定人体活动范围。",
                  input_archives=[dict(path=str(p), sha256=digest(p)) for p in (anatomical_path, fk_path)],
                  synthetic_plot=str(plot_path.resolve()), synthetic_joints=synthetic,
                  actual_right_knee=actual)
    (args.output / "joint_axes.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
