"""Export every prepared manifest shot with its complete registered processing PTS.

Uses rebuild_scope.json to check selected shot IDs, source order and full shot
ranges. A nine-shot input covers source 22.120 through 188.680 seconds; excluded
physical footage stays excluded. Subsets are recorded as subsets. This entry
shares the historical six-shot exporter and never optimizes numeric state.
Use --allow-sparse-observations to retain real frames without reliable points.
Use --native-library explicitly when the state was produced by an override DLL.
"""
from export_native_six_review import export as _export, main as _main


def export(run_root, state_path, output, label, compact_parameters=None, *,
           allow_sparse_observations=False, native_library=None):
    options = dict(legacy_six=False)
    if allow_sparse_observations:
        options["allow_sparse_observations"] = True
    if native_library is not None:
        options["native_library"] = native_library
    return _export(run_root, state_path, output, label, compact_parameters, **options)


def main():
    _main(legacy_six=False, description=__doc__)


if __name__ == "__main__":
    main()
