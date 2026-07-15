"""Native transport availability boundary."""


class NativeBackendUnavailable(RuntimeError):
    """Raised when the Embree extension has not been built."""


try:
    from ._native import EmbreeTransportScene  # type: ignore[import-not-found]
except ModuleNotFoundError:
    EmbreeTransportScene = None  # type: ignore[assignment,misc]


def require_native_backend() -> object:
    """Return the native class or fail explicitly."""

    if EmbreeTransportScene is None:
        raise NativeBackendUnavailable(
            "radcounter.radiation.native is not built; "
            "install Embree 4 and build the Isaac C++ extension"
        )
    return EmbreeTransportScene
