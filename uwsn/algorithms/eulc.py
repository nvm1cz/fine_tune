from .protocols import ExistingProtocolAdapter
from ..optimizers.eulc import run_eulc_cluster_head_selection


class EULCProtocol(ExistingProtocolAdapter):
    algorithm_id = "eulc"
    base_protocol = "eulc"
    implementation_status = "partial"
    selection_function = staticmethod(run_eulc_cluster_head_selection)
