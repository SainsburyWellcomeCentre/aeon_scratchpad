"""Stream classes for ONIX electrophysiology devices (Neuropixels 2.0 headstages).

Follows the layout of the ephys streams in aeon_mecha (``OnixClock``, a ``Bno055`` group,
headstage groups carrying ``HarpSync``) and lists only the timing streams QC needs. A
device is built from the headstage group plus the per-probe and ``Bno055`` groups, for
example ``Device("NeuropixelsV2Beta", NeuropixelsV2Beta, ProbeA, ProbeB, Bno055)``.

Every ``Clock`` stream is the ONIX acquisition clock stamped on each data frame by the
breakout board. All of them share one counter. ``HubSyncCounter`` is the headstage
hub's own clock captured with each probe sample; the BNO055 frames carry none.

Clock readers are ``swc.aeon.io.reader.Binary`` tagged with a ``uniform`` attribute, the
same idiom as ``expected_hz`` on continuous Harp streams: ``run_qc`` dispatches
``onix_clock_sequence`` on any ``Binary`` reader that carries it, and the value says
whether an irregular sample interval counts as an anomaly.
"""

import numpy as np
import swc.aeon.io.reader as _reader
from swc.aeon.schema.streams import Stream, StreamGroup

import aeon_qc.reader as _onix_reader


def clock_reader(pattern: str, uniform: bool = True) -> _reader.Binary:
    """Return a ``Binary`` reader for a ``uint64`` ONIX clock stream, tagged with ``uniform``."""
    reader = _reader.Binary(pattern, columns=("clock",), dtype=np.uint64)
    reader.uniform = uniform  # pyright: ignore[reportAttributeAccessIssue]
    return reader


class OnixClock(Stream):
    """Per-sample ONIX acquisition clock ticks of one data stream."""

    def __init__(self, pattern, uniform=True):
        """Initializes the OnixClock stream."""
        super().__init__(clock_reader(pattern, uniform=uniform))


class HubSyncCounter(Stream):
    """Per-sample headstage hub clock ticks of one data stream."""

    def __init__(self, pattern):
        """Initializes the HubSyncCounter stream."""
        super().__init__(_reader.Binary(pattern, columns=("hub_clock",), dtype=np.uint64))


class HarpSync(Stream):
    """ONIX to Harp synchronisation records, once per Harp second."""

    def __init__(self, pattern):
        """Initializes the HarpSync stream."""
        super().__init__(_onix_reader.HarpSync(f"{pattern}_HarpSync_*"))


class ProbeA(StreamGroup):
    """Timing streams of probe A: acquisition clock and headstage hub clock per sample."""

    def __init__(self, path):
        """Initializes the ProbeA stream group."""
        super().__init__(path)

    class ProbeAClock(OnixClock):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_ProbeA_Clock_*")

    class ProbeAHubSyncCounter(HubSyncCounter):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_ProbeA_HubSyncCounter_*")


class ProbeB(StreamGroup):
    """Timing streams of probe B: acquisition clock and headstage hub clock per sample."""

    def __init__(self, path):
        """Initializes the ProbeB stream group."""
        super().__init__(path)

    class ProbeBClock(OnixClock):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_ProbeB_Clock_*")

    class ProbeBHubSyncCounter(HubSyncCounter):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_ProbeB_HubSyncCounter_*")


class Bno055(StreamGroup):
    """Timing stream of the BNO055 orientation sensor, sampled at an irregular interval."""

    def __init__(self, path):
        """Initializes the Bno055 stream group."""
        super().__init__(path)

    class Bno055Clock(OnixClock):
        def __init__(self, pattern):
            super().__init__(f"{pattern}_Bno055_Clock_*", uniform=False)


class NeuropixelsV2(StreamGroup):
    """HarpSync stream of a Neuropixels 2.0e headstage as logged by the ephys workflows."""

    def __init__(self, path):
        """Initializes the NeuropixelsV2 stream group."""
        super().__init__(path, HarpSync)


class NeuropixelsV2Beta(StreamGroup):
    """HarpSync stream of a Neuropixels 2.0e beta headstage as logged by the ephys workflows."""

    def __init__(self, path):
        """Initializes the NeuropixelsV2Beta stream group."""
        super().__init__(path, HarpSync)
