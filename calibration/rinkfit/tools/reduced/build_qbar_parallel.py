"""Build an ordered OpenMP scheduler around the existing independent qbar rows.

The serial source, frozen headers, model and row formulas remain independent
inputs. Only row scheduling and ownership of output records change.
"""
from pathlib import Path
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Serial qbar scheduler signature changed: " + old[:100])
    return text.replace(old, new, 1)


def parallel_source(serial):
    prefix = serial.split("int main(int argc,char**argv)", 1)[0]
    text = "#include <omp.h>\n#include <atomic>\n#include <cstdlib>\n" + serial
    text = replace_once(text,
        ' for(std::size_t ordinal=0;ordinal<selected.size();++ordinal){const auto row=selected[ordinal];if(row>=n)throw std::runtime_error("qbar source row range");',
        ''' for(auto row:selected)if(row>=n)throw std::runtime_error("qbar source row range");
 int threads=20;if(const auto* setting=std::getenv("RINKFIT_QBAR_THREADS")){threads=std::stoi(setting);if(threads<1||threads>20)throw std::runtime_error("qbar CPU threads must be in [1,20]");}
 omp_set_dynamic(0);std::atomic<int> active_tasks{0},peak_tasks{0};std::vector<std::uint64_t> worker_tasks(threads,0);
 #pragma omp parallel for ordered schedule(dynamic,1) num_threads(threads) reduction(+:refined_stations)
 for(std::int64_t task=0;task<static_cast<std::int64_t>(selected.size());++task){const auto ordinal=static_cast<std::size_t>(task);const auto row=selected[ordinal];
 const int active_now=active_tasks.fetch_add(1)+1;int peak=peak_tasks.load();while(peak<active_now&&!peak_tasks.compare_exchange_weak(peak,active_now)){}++worker_tasks[omp_get_thread_num()];
 Json visibility_record,vertex_record,diagnosis_record;''')
    text = replace_once(text, "visibility_releases.push_back(evidence);", "visibility_record=evidence;")
    text = replace_once(text,
        'if(vertex_refinements.size()<100){evidence["source_row"]=row;vertex_refinements.push_back(evidence);}',
        'evidence["source_row"]=row;vertex_record=evidence;')
    text = replace_once(text,
        'if(diagnoses.size()<100)diagnoses.push_back({{"source_row",row},{"status",st},{"reason",e.what()},{"mode",mode[row]},{"stationarity_q_s",std::isfinite(qs)?Json(qs):Json(nullptr)}});',
        'diagnosis_record={{"source_row",row},{"status",st},{"reason",e.what()},{"mode",mode[row]},{"stationarity_q_s",std::isfinite(qs)?Json(qs):Json(nullptr)}};')
    root = 'std::isfinite(q)?model.project_rs(int(packet.model_material(row)),int(path[row]),s[row],(packet.pts[row]-256256)/12800.,v[row]).row:v[row]'
    text = replace_once(text, "  for(const auto& [b,c,g]:derivative)qcsr.add(b,c,g);",
        "  const double final_row=" + root + ";active_tasks.fetch_sub(1);\n"
        "  #pragma omp ordered\n  {\n"
        '   if(!visibility_record.is_null())visibility_releases.push_back(visibility_record);\n'
        '   if(!vertex_record.is_null()&&vertex_refinements.size()<100)vertex_refinements.push_back(vertex_record);\n'
        '   if(!diagnosis_record.is_null()&&diagnoses.size()<100)diagnoses.push_back(diagnosis_record);\n'
        "  for(const auto& [b,c,g]:derivative)qcsr.add(b,c,g);")
    text = replace_once(text, "linearization_v.push_back(" + root + ");", "linearization_v.push_back(final_row);")
    text = replace_once(text,
        ' }\n const double q_seconds=',
        '  }\n }\n const double q_seconds=')
    text = replace_once(text, ' report["visibility_active_set_releases"]=',
        ' report["CPU_row_scheduler"]={{"kind","OpenMP ordered independent ordinal"},{"requested_threads",threads},{"worker_tasks",worker_tasks},{"peak_concurrent_rows",peak_tasks.load()},{"row_output_order_preserved",true}};\n report["visibility_active_set_releases"]=')
    if text[ text.index(prefix):text.index(prefix)+len(prefix)] != prefix:
        raise AssertionError("Original qbar function prefix changed")
    return text, hashlib.sha256(prefix.encode()).hexdigest()


def build(serial_directory, output):
    serial_directory, output = Path(serial_directory).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = serial_directory / "qbar_finite_refine_v7.cpp"
    recipe = json.loads((serial_directory / "recipe.json").read_text(encoding="utf8"))
    text, core_sha = parallel_source(source.read_text(encoding="utf8"))
    output.mkdir(parents=True)
    shutil.copytree(serial_directory / "frozen_include", output / "frozen_include")
    shutil.copyfile(serial_directory / "event_model.cpp", output / "event_model.cpp")
    generated = output / source.name
    generated.write_text(text, encoding="utf8")
    command = [str(v).replace(str(serial_directory), str(output)) for v in recipe["command"]]
    command.insert(1, "/openmp")
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location("_qbar_build_environment", root / "calibration/rinkfit/cpp/gpu_rows/build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    started = time.monotonic()
    result = subprocess.run(command, cwd=output, env=module.environment(), capture_output=True, text=True, errors="replace")
    (output / "build.log").write_text(result.stdout + result.stderr, encoding="utf8")
    if result.returncode:
        raise RuntimeError("qbar OpenMP build failed: " + str(output / "build.log"))
    recipe.update(source_sha256=sha(generated), command=command,
        serial_reference=dict(directory=str(serial_directory), source_sha256=sha(source)),
        original_function_prefix_sha256=core_sha, scheduling_only=True, max_CPU_threads=20,
        build_seconds=time.monotonic()-started, executable_sha256=sha(output / "qbar_sparse.exe"))
    recipe["event_model_source"] = dict(path=str(output / "event_model.cpp"), sha256=sha(output / "event_model.cpp"))
    recipe["event_model_object"] = dict(path=str(output / "event_model.obj"), sha256=sha(output / "event_model.obj"))
    recipe["frozen_sources"] = {str(output / Path(v["file"])): v for v in recipe["frozen_sources"].values()}
    (output / "recipe.json").write_text(json.dumps(recipe, indent=2)+"\n", encoding="utf8")
    print(json.dumps(dict(output=str(output), build_seconds=recipe["build_seconds"], no_Solve=True)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial-qbar", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    build(args.serial_qbar, args.output)
