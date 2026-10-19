$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/OneDrive/codex_projs/RinkSpace'
$taskPython = 'D:/CodexData/RinkSpace_program_tracking_tmp/gtsam-env/python.exe'
$taskTracking = "$taskRoot/three_frame_refit/program_tracking"
$taskTmp = 'D:/CodexData/RinkSpace_program_tracking_tmp'
Set-Location -LiteralPath $taskRoot
$taskArguments = @(
    "$taskTracking/resume_source_identity_repair.py",
    '--resume-source', "$taskTmp/continuous_camera_model_v13_source_identity/main_long_side_curve_fix",
    '--camera', 'main_long_side',
    '--warm-group', "$taskTmp/continuous_camera_model_v12_complete_motion/stopped_before_source_identity_repair",
    '--warm-state', 'accepted_states.npz',
    '--prior-state-cache', "$taskTmp/continuous_camera_model_v4_domain_preflight/main_long_side",
    '--output', "$taskTmp/continuous_camera_model_v13_source_identity/main_long_side_crash_resume",
    '--rail-review', "$taskTmp/far_rail_reviewed_source_v1/far_rail_reviewed_source_selection.json",
    '--rail-review-sha256', '61eb15d384d43c16959375c79f6d560d78b8fc291eefaeea7354644e8210fd5b',
    '--motion-sidecar', "$taskTmp/continuous_source_motion_requalification_v2/main_hint0_source_review_pending.json",
    '--motion-sidecar-sha256', '345da13109dd394288d4de127575d755743d6e784c5dfba4eb12d00b550296e5',
    '--motion-review', "$taskRoot/evidence/round10_tail_user_issue_20260928/main_hint0_source_review_v2/agent_reviewed_motion_tail_v2_union.json",
    '--motion-review-sha256', '2083502f51c97314241ad1ca17ad7a81226b854f22f1993514dcb771dd086ea9',
    '--native-board-first-hit', '--native-central-domain', '--central-edge-brown15',
    '--native-source-edges', '--native-edge-threads', '8',
    '--reference-curve-workers', '8', '--native-trial-values', '--native-affine-priors',
    '--adaptive-source-curves', '--implicit-adaptive-curve-jacobian',
    '--iterations', '80'
)
& $taskPython @taskArguments
exit $LASTEXITCODE

