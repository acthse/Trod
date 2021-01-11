"""
    helo.model
    ~~~~~~~~~~
"""
from helo.model import core

JOINTYPE = core.JOINTYPE
ROWTYPE = core.ROWTYPE


def __getattr__(name: str) -> core.ModelType:
    if name == "Model":
        return core.ModelType(name, (core.ModelBase,), {})

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
