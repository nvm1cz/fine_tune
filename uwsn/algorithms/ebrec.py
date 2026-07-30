from .protocols import ExistingProtocolAdapter
from ..optimizers.ebrec import run_ebrec_cluster_head_selection


class EBRECProtocol(ExistingProtocolAdapter):
    algorithm_id = "ebrec"
    implementation_status = "partial"
    selection_function = staticmethod(run_ebrec_cluster_head_selection)
