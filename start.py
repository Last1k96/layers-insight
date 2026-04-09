#!/usr/bin/env python3
"""Cross-platform bootstrap: sets up Node.js, Python venv, deps, builds
the frontend, then launches the backend.  Works on Linux and Windows."""

import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

NODE_VERSION = "20.19.0"
NODE_DIR = Path(".node")
NODE_MARKER = NODE_DIR / ".version_marker"
IS_WINDOWS = sys.platform == "win32"


# ── helpers ──────────────────────────────────────────────────────────────

def needs_update(trigger: Path, marker: Path) -> bool:
    """True when *marker* is missing or *trigger* is newer."""
    return not marker.exists() or trigger.stat().st_mtime > marker.stat().st_mtime


def touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def run(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    shell = IS_WINDOWS and str(cmd[0]).endswith((".cmd", ".bat"))
    subprocess.run(cmd, cwd=cwd, env=env, check=True, shell=shell)


def any_file_newer_than(directory: Path, marker: Path) -> bool:
    if not marker.exists():
        return True
    threshold = marker.stat().st_mtime
    for p in directory.rglob("*"):
        if p.is_file() and p.stat().st_mtime > threshold:
            return True
    return False


# ── platform-aware paths ─────────────────────────────────────────────────

def venv_python() -> Path:
    if IS_WINDOWS:
        return Path(".venv", "Scripts", "python.exe")
    return Path(".venv", "bin", "python")


def node_bin_dir() -> Path:
    if IS_WINDOWS:
        return NODE_DIR
    return NODE_DIR / "bin"


def npm_cmd() -> Path:
    if IS_WINDOWS:
        return NODE_DIR / "npm.cmd"
    return NODE_DIR / "bin" / "npm"


# ── Node.js install ─────────────────────────────────────────────────────

def _node_arch() -> str:
    machine = platform.machine().lower()
    mapping = {
        "x86_64": "x64", "amd64": "x64",
        "aarch64": "arm64", "arm64": "arm64",
        "armv7l": "armv7l",
    }
    arch = mapping.get(machine)
    if arch is None:
        sys.exit(f"Unsupported architecture: {platform.machine()}")
    return arch


def ensure_local_node() -> None:
    if NODE_MARKER.exists() and NODE_MARKER.read_text().strip() == NODE_VERSION:
        return

    arch = _node_arch()
    if IS_WINDOWS:
        name = f"node-v{NODE_VERSION}-win-{arch}"
        filename = f"{name}.zip"
    else:
        name = f"node-v{NODE_VERSION}-linux-{arch}"
        filename = f"{name}.tar.xz"

    url = f"https://nodejs.org/dist/v{NODE_VERSION}/{filename}"
    print(f"Installing local Node.js v{NODE_VERSION}...")

    if NODE_DIR.exists():
        shutil.rmtree(NODE_DIR)
    NODE_DIR.mkdir(parents=True)

    with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        if IS_WINDOWS:
            _extract_zip(tmp_path, name)
        else:
            _extract_tar(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    NODE_MARKER.write_text(NODE_VERSION)
    print(f"Node.js v{NODE_VERSION} installed in {NODE_DIR}/")


def _extract_tar(archive: Path) -> None:
    """Extract .tar.xz, stripping the top-level directory."""
    with tarfile.open(archive, "r:xz") as tar:
        for member in tar.getmembers():
            # Strip first path component (e.g. "node-v20.19.0-linux-x64/...")
            parts = Path(member.name).parts
            if len(parts) <= 1:
                continue
            member.name = str(Path(*parts[1:]))
            tar.extract(member, NODE_DIR, filter="data")


def _extract_zip(archive: Path, top_dir: str) -> None:
    """Extract .zip, stripping the top-level directory."""
    with zipfile.ZipFile(archive) as zf:
        prefix = top_dir + "/"
        for info in zf.infolist():
            if not info.filename.startswith(prefix):
                continue
            rel = info.filename[len(prefix):]
            if not rel:
                continue
            dest = NODE_DIR / rel
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)


# ── --ov-path handling ───────────────────────────────────────────────────

def setup_ov_path(args: list[str]) -> None:
    prev = None
    for arg in args:
        if prev == "--ov-path":
            ov_resolved = str(Path(arg).resolve())

            # Library search path
            if IS_WINDOWS:
                env_var = "PATH"
            else:
                env_var = "LD_LIBRARY_PATH"
            current = os.environ.get(env_var, "")
            paths = current.split(os.pathsep) if current else []
            if ov_resolved not in paths:
                os.environ[env_var] = (
                    ov_resolved + os.pathsep + current if current else ov_resolved
                )

            # Python bindings
            python_dir = Path(ov_resolved) / "python"
            if python_dir.is_dir():
                py_path = os.environ.get("PYTHONPATH", "")
                py_dirs = py_path.split(os.pathsep) if py_path else []
                if str(python_dir) not in py_dirs:
                    os.environ["PYTHONPATH"] = (
                        str(python_dir) + os.pathsep + py_path if py_path else str(python_dir)
                    )
            break
        prev = arg


# ── main ─────────────────────────────────────────────────────────────────

def main() -> None:
    os.chdir(Path(__file__).resolve().parent)

    # Quick exit for --help (skip all setup)
    if "--help" in sys.argv[1:] or "-h" in sys.argv[1:]:
        python = str(venv_python()) if venv_python().exists() else sys.executable
        result = subprocess.run([python, "-m", "backend.main"] + sys.argv[1:])
        sys.exit(result.returncode)

    # 1. Local Node.js
    ensure_local_node()
    env = os.environ.copy()
    env["PATH"] = str(Path.cwd() / node_bin_dir()) + os.pathsep + env.get("PATH", "")

    # 2. Python venv
    if not Path(".venv").is_dir():
        print("Creating Python virtual environment...")
        run([sys.executable, "-m", "venv", ".venv"])

    # 3. Python deps
    if needs_update(Path("pyproject.toml"), Path(".venv/.deps_marker")):
        print("Installing Python dependencies...")
        run([str(venv_python()), "-m", "pip", "install", "-e", ".[dev]", "--quiet"])
        touch(Path(".venv/.deps_marker"))

    # 4. Root node_modules (elkjs)
    npm = str(npm_cmd())
    if needs_update(Path("package.json"), Path("node_modules/.deps_marker")):
        print("Installing root Node dependencies (elkjs)...")
        try:
            run([npm, "ci", "--quiet"], env=env)
        except subprocess.CalledProcessError:
            run([npm, "install", "--quiet"], env=env)
        touch(Path("node_modules/.deps_marker"))

    # 5. Frontend node_modules
    if needs_update(Path("frontend/package.json"), Path("frontend/node_modules/.deps_marker")):
        print("Installing frontend dependencies...")
        try:
            run([npm, "ci", "--quiet"], cwd=Path("frontend"), env=env)
        except subprocess.CalledProcessError:
            run([npm, "install", "--quiet"], cwd=Path("frontend"), env=env)
        touch(Path("frontend/node_modules/.deps_marker"))

    # 6. Frontend build
    build_marker = Path("frontend/dist/.build_marker")
    if not build_marker.exists() or any_file_newer_than(Path("frontend/src"), build_marker):
        print("Building frontend...")
        run([npm, "run", "build"], cwd=Path("frontend"), env=env)
        touch(build_marker)

    # 7. --ov-path environment setup
    setup_ov_path(sys.argv[1:])

    # 8. Launch
    python = str(venv_python())
    cmd = [python, "-m", "backend.main"] + sys.argv[1:]
    if IS_WINDOWS:
        result = subprocess.run(cmd)
        sys.exit(result.returncode)
    else:
        os.execvp(python, cmd)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}", file=sys.stderr)
        sys.exit(e.returncode or 1)
