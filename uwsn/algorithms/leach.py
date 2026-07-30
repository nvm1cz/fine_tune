from .protocols import ExistingProtocolAdapter
from ..optimizers.leach import run_leach_cluster_head_selection


class LEACHProtocol(ExistingProtocolAdapter):
    algorithm_id = "leach"
    implementation_status = "partial"
    uses_eulc_candidates = False
    selection_function = staticmethod(run_leach_cluster_head_selection)
