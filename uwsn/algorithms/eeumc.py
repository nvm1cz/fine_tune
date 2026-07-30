from .protocols import ExistingProtocolAdapter
from ..optimizers.eeumc import run_eeumc_cluster_head_selection


class EEUMCProtocol(ExistingProtocolAdapter):
    algorithm_id = "eeumc"
    implementation_status = "partial"
    selection_function = staticmethod(run_eeumc_cluster_head_selection)
