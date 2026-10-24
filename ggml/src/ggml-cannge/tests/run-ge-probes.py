"""Run isolated GE matmul diagnostics on an already configured Ascend host. No network operations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


CASES = [
    ("control", 16, 256),
    ("long-k", 16, 2560),
    ("wide-n", 4096, 256),
    ("projection", 4096, 2560),
]


def run(command, directory, log, env, timeout):
    started = time.monotonic()
    timed_out = False
    with log.open("wb") as output:
        process = subprocess.Popen(command, cwd=directory, env=env, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            code = 124
        finally:
            # Only this command's process group. Failed GE processes can leave compiler children.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    return {"command": command, "exit_code": code, "timed_out": timed_out,
            "seconds": round(time.monotonic() - started, 3), "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new directory outside the checkout")
    parser.add_argument("--sdk", type=Path, default=os.environ.get("ASCEND_TOOLKIT_HOME"))
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--operator", choices=("bmm", "matmul", "matmul-v2"), default="bmm")
    parser.add_argument("--plan", action="store_true", help="print cases without building or touching hardware")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps([{"case": name, "operator": args.operator, "M": 16, "N": n, "K": k}
                          for name, n, k in CASES], indent=2))
        return 0
    if sys.platform != "linux" or not args.sdk or not args.output or args.timeout <= 0:
        parser.error("execution requires Linux, --sdk (or ASCEND_TOOLKIT_HOME), --output and a positive timeout")
    source = Path(__file__).resolve().with_name("ge-matmul-probe.cpp")
    root = source.parents[4]
    directory = args.output.resolve()
    if directory == root or root in directory.parents:
        parser.error("--output must be outside the source checkout")
    directory.mkdir(parents=True, exist_ok=False)
    sdk = args.sdk.resolve()
    env = os.environ.copy()
    # Known debug and backend settings must not silently change the experiment.
    for key in list(env):
        if key.startswith(("GGML_CANNGE_", "DUMP_")) or key in (
                "NPU_COLLECT_PATH", "ASCEND_WORK_PATH", "LD_PRELOAD", "IGNORE_INFER_ERROR"):
            env.pop(key)
    results = {"probe_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "sdk": str(sdk), "operator": args.operator, "cases": [], "timeout_seconds": args.timeout}
    report = directory / "results.json"

    def save():
        report.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    save()
    # Local provenance only; neither command fetches nor alters the checkout.
    results["revision"] = run(["git", "-C", str(root), "rev-parse", "HEAD"], directory,
                              directory / "revision.log", env, 15)
    results["worktree"] = run(["git", "-C", str(root), "status", "--short"], directory,
                              directory / "worktree.log", env, 15)
    results["device"] = run(["npu-smi", "info"], directory, directory / "device.log", env, 30)
    save()
    if results["device"]["exit_code"] != 0:
        return 1
    binary = directory / "ge-matmul-probe"
    command = [args.compiler, "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", str(source)]
    command += ["-I" + str(sdk / suffix) for suffix in ("include", "include/es", "include/ge")]
    command += ["-L" + str(sdk / "lib64"), "-lascendcl", "-lge_runner", "-lgraph", "-lgraph_base",
                "-les_math", "-les_nn", "-leager_style_graph_builder_base", "-lc_sec", "-o", str(binary)]
    results["build"] = run(command, directory, directory / "build.log", env, 300)
    save()
    if results["build"]["exit_code"] != 0:
        return 1
    for name, n, k in CASES:
        case_dir = directory / name
        case_dir.mkdir()
        case_env = dict(env, DUMP_GE_GRAPH="2", DUMP_GRAPH_LEVEL="3", DUMP_GRAPH_FORMAT="ge_proto",
                        DUMP_GRAPH_PATH=str(case_dir), NPU_COLLECT_PATH=str(case_dir),
                        ASCEND_GLOBAL_LOG_LEVEL="1", ASCEND_SLOG_PRINT_TO_STDOUT="1")
        result = run([str(binary), args.operator, str(n), str(k)], case_dir, case_dir / "run.log",
                     case_env, args.timeout)
        result.update(case=name, M=16, N=n, K=k)
        log_text = (case_dir / "run.log").read_text(errors="replace")
        result["passed"] = result["exit_code"] == 0 and f"PASS: all {16*n} outputs equal {k}" in log_text
        results["cases"].append(result)
        save()
        print(f"{name}: {'PASS' if result['passed'] else 'FAIL'} (exit {result['exit_code']})", flush=True)
        # A failed control or a hang invalidates further batch execution. Ordinary probe failures are recorded.
        if result["timed_out"] or (name == "control" and not result["passed"]):
            break
    results["device_after"] = run(["npu-smi", "info"], directory, directory / "device-after.log", env, 30)
    save()
    return 0 if len(results["cases"]) == len(CASES) and all(c["passed"] for c in results["cases"]) else 1


if __name__ == "__main__":
    sys.exit(main())
