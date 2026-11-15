"""Prepare the same GGUF and llama.cpp runtime in a Kaggle GPU session."""

import argparse
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from prepare_assets import model_assets

LLAMA_VERSION = "v1.0.0"


def server_environment(server):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["LLAMA_SERVER"] = str(server)
    lib_dirs = [str(server.parent)]
    if env.get("LD_LIBRARY_PATH"):
        lib_dirs.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = ":".join(lib_dirs)
    return env


def probe(server):
    if not server.is_file():
        return False
    server.chmod(server.stat().st_mode | 0o111)
    try:
        check = subprocess.run([str(server), "--version"], env=server_environment(server),
                               capture_output=True, text=True, timeout=20, check=False)
        if check.returncode == 0:
            return True
        print(f"Cannot use {server}: {check.stderr.strip()[-600:]}", flush=True)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"Cannot use {server}: {exc}", flush=True)
    return False


def restore_library_links(folder):
    for library in sorted(folder.glob("lib*.so.*.*")):
        if library.is_symlink() or not library.is_file():
            continue
        with library.open("rb") as stream:
            if stream.read(4) != b"\x7fELF":
                continue
        stem, version = library.name.split(".so.", 1)
        major = folder / f"{stem}.so.{version.split('.')[0]}"
        for link, target in [(major, library), (folder / f"{stem}.so", major)]:
            if link.is_file() and not link.is_symlink():
                with link.open("rb") as stream:
                    if stream.read(4) == b"\x7fELF":
                        continue
            link.unlink(missing_ok=True)
            link.symlink_to(target.name)


def cached_server(work, inputs):
    for server in sorted(work.glob("llama-cache-*/**/llama-server")):
        if probe(server):
            return server
    sources = [
        *sorted(inputs.rglob("llama_cuda_t4.tar.gz")),
        *sorted(inputs.rglob("bin/llama-server")),
    ]
    for source in sources:
        cache = Path(tempfile.mkdtemp(prefix="llama-cache-", dir=work))
        try:
            if source.name.endswith(".tar.gz"):
                with tarfile.open(source, "r:gz") as tar:
                    tar.extractall(cache, filter="data")
            else:
                shutil.copytree(source.parent, cache / "bin", symlinks=True)
            for server in sorted(cache.rglob("llama-server")):
                restore_library_links(server.parent)
                if probe(server):
                    print(f"Using cached llama-server from {source}", flush=True)
                    return server
        except (OSError, tarfile.TarError) as exc:
            print(f"Cannot restore cache {source}: {exc}", flush=True)
        shutil.rmtree(cache)
    return None


def compile_server(work, architectures):
    repository = work / "llama.cpp"
    build = work / "llama-build"
    server = build / "bin" / "llama-server"
    if server.is_file():
        return server
    if not repository.is_dir():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", LLAMA_VERSION,
                        "https://github.com/ggml-org/llama.cpp.git", str(repository)], check=True)
    subprocess.run(["cmake", "-S", str(repository), "-B", str(build), "-DGGML_CUDA=ON",
                    f"-DCMAKE_CUDA_ARCHITECTURES={architectures}", "-DGGML_NATIVE=OFF",
                    "-DGGML_CUDA_NO_VMM=ON", "-DGGML_CUDA_FA=OFF", "-DGGML_CUDA_NCCL=OFF",
                    "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TOOLS=ON", "-DLLAMA_BUILD_SERVER=ON"], check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "llama-server", "-j", "4"], check=True)
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=Path("/kaggle/working/datamate_runtime"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--server", type=Path)
    parser.add_argument("--arch", default="75")
    parser.add_argument("--inputs", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--prepare", type=Path, help="Write runtime paths and let the benchmark manage the process")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    try:
        gpu = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True)
        if "GPU " not in gpu.stdout:
            parser.error("Enable the Kaggle GPU accelerator (T4) before starting Qwen")
    except (OSError, subprocess.CalledProcessError):
        parser.error("Kaggle GPU is unavailable. Enable the GPU accelerator (T4).")
    weights = model_assets(args.model)
    server = args.server.resolve() if args.server else (
        cached_server(args.work, args.inputs) or compile_server(args.work, args.arch)
    )
    if not server.is_file():
        parser.error(f"llama-server not found: {server}")
    if args.prepare:
        args.prepare.write_text(json.dumps({"server": str(server), "model": str(weights)}), encoding="utf-8")
        return
    os.execvpe("bash", ["bash", str(Path(__file__).with_name("serve.sh")), str(weights)],
               server_environment(server))


if __name__ == "__main__":
    main()
