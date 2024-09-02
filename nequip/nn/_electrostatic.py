import math
from typing import List

import numpy as np
import torch
from ase.data import covalent_radii
from e3nn.o3 import Irreps, Linear

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin
from nequip.data._keys import CHARGES_KEY, ELECTROSTATIC_ENERGY_KEY, TOTAL_CHARGE_KEY

class SumEnergies(GraphModuleMixin, torch.nn.Module):
    def __init__(
        self,
        input_fields: List[str],
        out_field: str = AtomicDataDict.TOTAL_ENERGY_KEY,
        irreps_in=None,
    ):
        super().__init__()

        self.input_fields = input_fields
        self.out_field = out_field
        irreps_out = {self.out_field: irreps_in[self.out_field]}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[
                self.out_field,
            ]
            + input_fields,
            irreps_out=irreps_out,
        )

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        total = torch.zeros_like(data[self.out_field])
        for field in self.input_fields:
            total += data[field]
            # print('field, value:', field, data[field])
        data[self.out_field] = total

        return data
    