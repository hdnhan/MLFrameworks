from pathlib import Path

from invoke import task
from invoke.context import Context

folders_to_ignore = [
    "build/",
]


@task
def pyfmt(c: Context, verbose: bool = True) -> None:
    files = []
    files.extend(Path(".").resolve().rglob("*.py"))

    files_to_format = []
    for file in files:
        if any(folder in file.as_posix() for folder in folders_to_ignore):
            continue
        files_to_format.append(file.as_posix())

    options = ""
    if verbose:
        options += " --verbose"

    c.run(f"ruff format {options} --check {' '.join(files_to_format)}", echo=True)


@task
def pylint(c: Context, verbose: bool = True) -> None:
    files = []
    files.extend(Path(".").resolve().rglob("*.py"))

    files_to_format = []
    for file in files:
        if any(folder in file.as_posix() for folder in folders_to_ignore):
            continue
        files_to_format.append(file.as_posix())

    options = ""
    if verbose:
        options += " --verbose"
    c.run(f"ruff check {options} {' '.join(files_to_format)}", echo=True)


@task
def cppfmt(c: Context, check: bool = True, verbose: bool = True) -> None:
    # [ClangFormat docs](https://clang.llvm.org/docs/ClangFormat.html)
    # find . -type f -name "*.cpp" -o -name "*.hpp"  | xargs clang-format -style=file -i

    files = []
    files.extend(Path(".").resolve().rglob("*.cpp"))
    files.extend(Path(".").resolve().rglob("*.hpp"))
    files.extend(Path(".").resolve().rglob("*.h"))
    files.extend(Path(".").resolve().rglob("*.c"))
    files.extend(Path(".").resolve().rglob("*.m"))
    files.extend(Path(".").resolve().rglob("*.mm"))

    files_to_format = []
    for file in files:
        if any(folder in file.as_posix() for folder in folders_to_ignore):
            continue
        files_to_format.append(file.as_posix())

    options = "--dry-run -Werror" if check else "-i"
    if verbose:
        options += " --verbose"
    c.run(f"clang-format {options} -style=file:.clang-format {' '.join(files_to_format)}", echo=True)
