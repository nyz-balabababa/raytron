from pathlib import Path


ESAM_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ESAM_ROOT.parents[1]


def _coerce_path(path_like):
    if path_like is None:
        return None
    return path_like if isinstance(path_like, Path) else Path(path_like)


def resolve_project_path(path_like):
    path = _coerce_path(path_like)
    if path is None:
        return None
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def resolve_esam_path(path_like):
    path = _coerce_path(path_like)
    if path is None:
        return None
    return path if path.is_absolute() else (ESAM_ROOT / path).resolve()


def ensure_file(path_like, desc):
    path = resolve_project_path(path_like)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"{desc} not found: {path}. Please pass --{desc} explicitly.")
    return path


def ensure_dir(path_like, desc):
    path = resolve_project_path(path_like)
    if path is None or not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}. Please pass --{desc} explicitly.")
    if not path.is_dir():
        raise NotADirectoryError(f"{desc} is not a directory: {path}.")
    return path
