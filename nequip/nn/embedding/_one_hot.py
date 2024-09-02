import torch
import torch.nn.functional

from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode

from nequip.data import AtomicDataDict
from .._graph_mixin import GraphModuleMixin


@compile_mode("script")
class OneHotAtomEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]
        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        one_hot = torch.nn.functional.one_hot(
            type_numbers, num_classes=self.num_types
        ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot
        return data

@compile_mode("script")
class ChargeEncoding(GraphModuleMixin, torch.nn.Module):
    """Copmute a one-hot floating point encoding of atoms' discrete atom types.

    Args:
        set_features: If ``True`` (default), ``node_features`` will be set in addition to ``node_attrs``.
    """

    num_types: int
    set_features: bool

    def __init__(
        self,
        num_types: int,
        qmin: float,
        qmax: float,
        num_categories: int,
        set_features: bool = True,
        irreps_in=None,
    ):
        super().__init__()
        self.num_types = num_types
        self.set_features = set_features
        # Output irreps are num_types even (invariant) scalars
        irreps_out = {AtomicDataDict.NODE_ATTRS_KEY: Irreps([(self.num_types + num_categories, (0, 1))])}
        if self.set_features:
            irreps_out[AtomicDataDict.NODE_FEATURES_KEY] = irreps_out[
                AtomicDataDict.NODE_ATTRS_KEY
            ]

        #charge categories
        self.filter = torch.linspace(qmin, qmax, num_categories)
        self.var = (qmax - qmin) / num_categories

        self._init_irreps(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        device = data[AtomicDataDict.POSITIONS_KEY].device
        charge_vector = torch.exp(-((data[AtomicDataDict.CHARGES_KEY] - self.filter.to(device)) ** 2) / self.var ** 2)
#        with np.printoptions(threshold=float('inf')):
#            print(charge_vector.detach().numpy())
        # type_numbers = data[AtomicDataDict.ATOM_TYPE_KEY].squeeze(-1)
        # one_hot = torch.nn.functional.one_hot(
        #     type_numbers, num_classes=self.num_types
        # ).to(device=type_numbers.device, dtype=data[AtomicDataDict.POSITIONS_KEY].dtype)
        #one_hot_with_charge = torch.cat((data[AtomicDataDict.NODE_ATTRS_KEY], data[AtomicDataDict.CHARGES_KEY]), dim=1)
        one_hot_with_charge = torch.cat((data[AtomicDataDict.NODE_ATTRS_KEY], charge_vector), dim=1)
#        print('one_hot_with_charge shape = ', one_hot_with_charge.shape)
        data[AtomicDataDict.NODE_ATTRS_KEY] = one_hot_with_charge
        if self.set_features:
            data[AtomicDataDict.NODE_FEATURES_KEY] = one_hot_with_charge
        return data