import os
from pathlib import Path


def install_dependencies_format():
    os.system("pip install clang-format==18.1.8")
    os.system("pip install ruff==0.5.5")


folders_to_ignore = [
    "build/",
]


# [ClangFormat docs](https://clang.llvm.org/docs/ClangFormat.html)
# find . -type f -name "*.cpp" -o -name "*.hpp"  | xargs clang-format -style=file -i
def format_cpp_files():
    # find *.cpp, *.hpp, *.h, *.c, *.m, *.mm files
    files = []
    files.extend(Path(".").resolve().rglob("*.cpp"))
    files.extend(Path(".").resolve().rglob("*.hpp"))
    files.extend(Path(".").resolve().rglob("*.h"))
    files.extend(Path(".").resolve().rglob("*.c"))
    files.extend(Path(".").resolve().rglob("*.m"))
    files.extend(Path(".").resolve().rglob("*.mm"))

    # clang-format
    for file in files:
        if any(folder in file.as_posix() for folder in folders_to_ignore):
            continue

        print(f"Formatting {file}...")
        os.system(f"clang-format -i -style=file:.clang-format {file}")


def format_and_lint_python_files():
    # find *.py files
    files = []
    files.extend(Path(".").resolve().rglob("*.py"))

    # ruff
    for file in files:
        if any(folder in file.as_posix() for folder in folders_to_ignore):
            continue

        print(f"Formatting {file}...")
        # format
        os.system(f"ruff format {file}")
        # lint
        os.system(f"ruff check {file}")  # it's good to use it


if __name__ == "__main__":
    install_dependencies_format()
    format_cpp_files()
    format_and_lint_python_files()
