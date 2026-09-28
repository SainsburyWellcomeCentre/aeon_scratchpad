"""Readers for ONIX electrophysiology streams logged by the Aeon ephys workflows.

A reader takes the full file pattern, names its own columns and returns a DataFrame
from ``read(path)``. The per-sample clock streams need no class of their own; the
stream definitions in ``aeon_qc.onix`` use ``swc.aeon.io.reader.Binary`` directly.
"""

from swc.aeon.io.reader import Csv

NOMINAL_CLOCK_HZ = 250_000_000
"""Nominal rate of the ONIX acquisition clock that stamps every ONIX data frame, in Hz."""


class HarpSync(Csv):
    """Extracts the once-per-second ONIX to Harp synchronisation records.

    Each row pairs the ONIX acquisition clock (``clock``) and headstage hub clock
    (``hub_clock``) captured on a heartbeat with the Harp second it encodes
    (``harp_time``). The index is the ``Seconds`` column written by the workflow.
    """

    def __init__(self, pattern: str):
        """Initialize the object with the specified pattern."""
        super().__init__(pattern, columns=("clock", "hub_clock", "harp_time"))
