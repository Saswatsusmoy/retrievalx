from __future__ import annotations

import importlib.machinery
import importlib.util
import site
import sys
from pathlib import Path
from types import ModuleType


def _load_extension() -> ModuleType:
    candidate_roots = [Path(__file__).parent]
    candidate_roots.extend(Path(p) / "retrievalx" for p in site.getsitepackages())
    candidate_roots.append(Path(site.getusersitepackages()) / "retrievalx")

    for package_dir in candidate_roots:
        if not package_dir.exists():
            continue

        for pattern in ("_native*.so", "_native*.pyd", "_native*.dylib"):
            for candidate in package_dir.glob(pattern):
                loader = importlib.machinery.ExtensionFileLoader(__name__, str(candidate))
                spec = importlib.util.spec_from_file_location(__name__, candidate, loader=loader)
                if spec is None or spec.loader is None:
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[__name__] = module
                spec.loader.exec_module(module)
                return module

    raise ImportError(
        "retrievalx native extension not found. Build/install wheel with maturin first."
    )


_module = _load_extension()
NativeBM25Index = _module.NativeBM25Index


def __getattr__(name: str) -> object:
    return getattr(_module, name)
