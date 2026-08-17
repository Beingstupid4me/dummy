import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_cmd(args: list[str], env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    result = subprocess.run(args, capture_output=True, text=True, env=env, cwd=cwd)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(stderr)


def build_env_with_pandoc() -> dict[str, str]:
    env = os.environ.copy()
    if shutil.which("pandoc"):
        return env

    candidate_paths = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Pandoc" / "pandoc.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Pandoc" / "pandoc.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Pandoc" / "pandoc.exe",
    ]

    for candidate in candidate_paths:
        if candidate.is_file():
            env["PATH"] = f"{candidate.parent};{env.get('PATH', '')}"
            return env

    return env


def build_env_with_tex(env: dict[str, str]) -> dict[str, str]:
    if shutil.which("xelatex"):
        return env

    candidate_dirs = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64",
        Path(os.environ.get("ProgramFiles", "")) / "MiKTeX" / "miktex" / "bin" / "x64",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "MiKTeX" / "miktex" / "bin" / "x64",
    ]

    for candidate in candidate_dirs:
        if (candidate / "xelatex.exe").is_file():
            env["PATH"] = f"{candidate};{env.get('PATH', '')}"
            return env

    return env


def export_notebook(notebook_path: Path, output_path: Path, execute: bool) -> None:
    output_dir = output_path.parent
    output_name = output_path.stem
    env = build_env_with_tex(build_env_with_pandoc())

    base_cmd = [
        "--output",
        output_name,
        "--output-dir",
        str(output_dir),
        str(notebook_path),
    ]

    if execute:
        base_cmd.insert(0, "--execute")

    runners = [
        [sys.executable, "-m", "jupyter", "nbconvert"],
        [sys.executable, "-m", "nbconvert"],
    ]

    last_error = None
    for runner in runners:
        try:
            run_cmd(runner + ["--to", "pdf"] + base_cmd, env=env)
            return
        except Exception as exc:
            last_error = exc

    # Try a manual LaTeX pipeline in the output directory for reliable image resolution.
    for runner in runners:
        try:
            run_cmd(runner + ["--to", "latex"] + base_cmd, env=env)
            tex_file = output_dir / f"{output_name}.tex"
            latex_cmd = ["xelatex", "-interaction=nonstopmode", "-file-line-error", str(tex_file.name)]
            run_cmd(latex_cmd, env=env, cwd=output_dir)
            run_cmd(latex_cmd, env=env, cwd=output_dir)
            return
        except Exception as exc:
            last_error = exc

    print("PDF export failed, trying webpdf as fallback...")
    print(f"Reason: {last_error}")

    for runner in runners:
        run_cmd(runner + ["--to", "webpdf"] + base_cmd, env=env)
        return


def find_notebook(cwd: Path) -> Path:
    notebooks = sorted(cwd.glob("*.ipynb"))
    if not notebooks:
        raise SystemExit(f"No .ipynb files found in {cwd}")

    if len(notebooks) == 1:
        return notebooks[0]

    newest = max(notebooks, key=lambda path: path.stat().st_mtime)
    print("Multiple notebooks found. Using the most recently modified one:")
    print(f"- {newest.name}")
    return newest


def main() -> None:
    cwd = Path.cwd()
    notebook_path = find_notebook(cwd)
    output_path = notebook_path.with_suffix(".pdf")
    export_notebook(notebook_path, output_path, execute=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
