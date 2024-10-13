from typing import Optional
import logging

from e3nn import o3
from e3nn.o3 import Irreps

from nequip.data import AtomicDataDict, AtomicDataset, register_fields
from nequip.nn import (
    SequentialGraphNetwork,
    AtomwiseLinear,
    AtomwiseReduce,
    ConvNetLayer,
)

from nequip.nn.embedding import (
    OneHotAtomEncoding, ChargeEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)

from . import builder_utils
from nequip.nn._ewald import EwaldQeq
from nequip.nn._electrostatic import SumEnergies, Qeq

def SimpleIrrepsConfig(config, prefix: Optional[str] = None):
    """Builder that pre-processes options to allow "simple" configuration of irreps."""

    # We allow some simpler parameters to be provided, but if they are,
    # they have to be correct and not overridden
    simple_irreps_keys = ["l_max", "parity", "num_features"]
    real_irreps_keys = [
        "chemical_embedding_irreps_out",
        "feature_irreps_hidden",
        "irreps_edge_sh",
        "conv_to_output_hidden_irreps_out",
    ]

    prefix = "" if prefix is None else f"{prefix}_"

    has_simple: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in simple_irreps_keys
    )
    has_full: bool = any(
        (f"{prefix}{k}" in config) or (k in config) for k in real_irreps_keys
    )
    assert has_simple or has_full

    update = {}
    if has_simple:
        # nothing to do if not
        lmax = config.get(f"{prefix}l_max", config["l_max"])
        parity = config.get(f"{prefix}parity", config["parity"])
        num_features = config.get(f"{prefix}num_features", config["num_features"])
        update[f"{prefix}chemical_embedding_irreps_out"] = repr(
            o3.Irreps([(num_features, (0, 1))])  # n scalars
        )
        update[f"{prefix}irreps_edge_sh"] = repr(
            o3.Irreps.spherical_harmonics(lmax=lmax, p=-1 if parity else 1)
        )
        update[f"{prefix}feature_irreps_hidden"] = repr(
            o3.Irreps(
                [
                    (num_features, (l, p))
                    for p in ((1, -1) if parity else (1,))
                    for l in range(lmax + 1)
                ]
            )
        )
        update[f"{prefix}conv_to_output_hidden_irreps_out"] = repr(
            # num_features // 2  scalars
            o3.Irreps([(max(1, num_features // 2), (0, 1))])
        )

    # check update is consistant with config
    # (this is necessary since it is not possible
    #  to delete keys from config, so instead of
    #  making simple and full styles mutually
    #  exclusive, we just insist that if full
    #  and simple are provided, full must be
    #  consistant with simple)
    for k, v in update.items():
        if k in config:
            assert (
                config[k] == v
            ), f"For key {k}, the full irreps options had value `{config[k]}` inconsistant with the value derived from the simple irreps options `{v}`"
        config[k] = v


def EnergyModel(
    config, initialize: bool, dataset: Optional[AtomicDataset] = None
) -> SequentialGraphNetwork:
    """Base default energy model archetecture.

    For minimal and full configuration option listings, see ``minimal.yaml`` and ``example.yaml``.
    """
    logging.debug("Start building the network model")

    builder_utils.add_avg_num_neighbors(
        config=config, initialize=initialize, dataset=dataset
    )

    energy_scale = config.get("_global_scale", 1.0)
    use_ewald_scaling = config.get("use_ewald_scaling", False)

    if use_ewald_scaling: # use ewald scaling
        energy_scale = energy_scale
    else: # do not use ewald scaling
        energy_scale = 1.0
    num_layers = config.get("num_layers", 3)
    num_layers_charge = config.get("num_layers_charge", 3)
    pbc = config.get("pbc", False)

    layers = {
        # -- Encode --
        "one_hot": OneHotAtomEncoding,
        "spharm_edges": SphericalHarmonicEdgeAttrs,
        "radial_basis": RadialBasisEdgeEncoding,
        # -- Embed features --
        "chemical_embedding": AtomwiseLinear,
    }

    # add convnet layers
    # insertion preserves order
    for layer_i in range(num_layers):
        layers[f"layer{layer_i}_convnet"] = ConvNetLayer
    
    
    layers["before_charge_prediction"] = AtomwiseLinear
    config['before_charge_prediction_irreps_out'] = repr(o3.Irreps([(config.get("num_features"), (0, 1))]))
                                         
    if pbc:
        layers["total_energy_with_qeq"] = (
                    EwaldQeq,
                    dict(scale=energy_scale), #energy_scale
                )
    else:
        layers["total_energy_with_qeq"] = (
                    Qeq,
                    dict(scale=energy_scale), #energy_scale
                )

    #Calculate the charges using Qeq
    layers["one_hot2"] = ChargeEncoding
    layers["spharm_edges2"] = SphericalHarmonicEdgeAttrs
    layers["radial_basis2"] = RadialBasisEdgeEncoding
    layers["chemical_charge_embedding"] = AtomwiseLinear
    config['chemical_charge_embedding_irreps_out'] = repr(o3.Irreps([(config.get("num_features"), (0, 1))]))

    for layer_i in range(num_layers_charge):
        layers[f"layer{layer_i}_charge_convnet"] = ConvNetLayer

    # .update also maintains insertion order
    layers.update(
        {
            # TODO: the next linear throws out all L > 0, don't create them in the last layer of convnet
            # -- output block --
            "conv_to_output_hidden": AtomwiseLinear,
            "output_hidden_to_scalar": (
                AtomwiseLinear,
                dict(irreps_out="1x0e", out_field=AtomicDataDict.PER_ATOM_ENERGY_KEY),
            ),
        }
    )

    layers["total_energy_sum"] = (
        AtomwiseReduce,
        dict(
            reduce="sum",
            field=AtomicDataDict.PER_ATOM_ENERGY_KEY,
            out_field=AtomicDataDict.TOTAL_ENERGY_KEY,
        ),
    )

    layers["sum_energy_terms"] = (
            SumEnergies,
            dict(
                input_fields=[AtomicDataDict.TOTAL_ENERGY_KEY, AtomicDataDict.ELECTROSTATIC_ENERGY_KEY],
            ),
        )

    # irreps_in = {
    #     TOTAL_CHARGE_KEY: Irreps("1x0e")
    # }

    return SequentialGraphNetwork.from_parameters(
        shared_params=config,
        layers=layers,
    )
