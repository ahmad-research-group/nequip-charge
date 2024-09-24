import math
from typing import List, Optional

import numpy as np
import torch
from ase.data import covalent_radii
from e3nn.o3 import Irreps, Linear
from scipy import constants

from nequip.data import AtomicDataDict
from nequip.nn import GraphModuleMixin

from nequip.utils.batch_ops import bincount

class Ewald(GraphModuleMixin, torch.nn.Module):
    """
    Layer to calculate electrostatic energy via Ewald summation
    """

    def __init__(
        self,
        atomic_numbers: List[int],  # automatically passed from shared_params 
        out_field: str = AtomicDataDict.ELECTROSTATIC_ENERGY_KEY,
        irreps_in=None,
        scale: float = 1.0,  # std of dataset, physical unit
    ):
        super().__init__()

        self.scale = scale
        

        self.out_field = out_field
        irreps_out = {self.out_field: Irreps("1x0e")}
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.POSITIONS_KEY, AtomicDataDict.CHARGES_KEY],
            irreps_out=irreps_out,
        )

        # sigma: species_index (0-indexed) -> covalent radius
        self.sigma = torch.from_numpy(np.array(covalent_radii[atomic_numbers]))
        if len(atomic_numbers) == 1:
            self.sigma = torch.unsqueeze(self.sigma, 0)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        device = data[AtomicDataDict.POSITIONS_KEY].device
        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        sigmas = self.sigma[species_idx].to(device)
        charges = data[AtomicDataDict.CHARGES_KEY]  # (num_atoms, 1)

        ptr = data["ptr"]
        batch_size = ptr.shape[0] - 1

        ele = torch.zeros(batch_size, device=device)
        for bi in range(batch_size):
            cell_bi = data[AtomicDataDict.CELL_KEY][bi]
            pos_bi = data[AtomicDataDict.POSITIONS_KEY][ptr[bi] : ptr[bi + 1]]
            sigmas_bi = sigmas[ptr[bi] : ptr[bi + 1]]
            ewald_bi = EwaldAuxiliary(
                cell=cell_bi,
                pos=pos_bi,
                sigmas=sigmas_bi,
                point_charge=False,
            )

            # EwaldAuxiliary.calc_energy takes only 1-dimensional tensor
            charges_bi = torch.squeeze(charges[ptr[bi] : ptr[bi + 1]], dim=1)
            ele[bi] = ewald_bi.calc_energy(charges_bi) / self.scale

        data[self.out_field] = torch.unsqueeze(ele, dim=1)  # (batch_size, 1)
        return data


class EwaldQeq(GraphModuleMixin, torch.nn.Module):
    """
    Qeq for periodic systems
    """

    def __init__(
        self,
        atomic_numbers: List[int],  # automatically passed from shared_params
        out_field: str = AtomicDataDict.ELECTROSTATIC_ENERGY_KEY,
        irreps_in=None,
        scale: float = 1.0,  # std of dataset, physical unit
    ):
        super().__init__()

        self.scale = scale
        print('ewald scaling factor = ', self.scale)
        self.out_field = out_field
        irreps_out = {
            self.out_field: Irreps("1x0e"),
            AtomicDataDict.CHARGES_KEY: Irreps("1x0e"),
        }
        self._init_irreps(
            irreps_in=irreps_in,
            required_irreps_in=[AtomicDataDict.POSITIONS_KEY, AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out=irreps_out,
        )

        # sigma: species_index (0-indexed) -> covalent radius
        covalent_radii_for_atoms = covalent_radii[atomic_numbers]
        self.sigma = torch.tensor([x for _, x in sorted(zip(atomic_numbers, covalent_radii_for_atoms))])
        if len(atomic_numbers) == 1:
            self.sigma = torch.unsqueeze(self.sigma, 0)

        self.to_chi = Linear(
            irreps_in=irreps_in[AtomicDataDict.NODE_FEATURES_KEY],
            irreps_out=Irreps("1x0e"),
        )
        hardness_params = torch.ones(len(atomic_numbers)) #randn?
        self.to_hardness = torch.nn.Parameter(data=hardness_params)

    def forward(self, data: AtomicDataDict.Type) -> AtomicDataDict.Type:
        device = data[AtomicDataDict.POSITIONS_KEY].device
        species_idx = data[AtomicDataDict.ATOM_TYPE_KEY]
        sigmas = torch.squeeze(self.sigma[species_idx].to(device), dim =1)
        chi = self.to_chi(data[AtomicDataDict.NODE_FEATURES_KEY])  # (num_atoms, 1)
        # square here to restrit hardness to be positive!
        hardness = torch.square(self.to_hardness[species_idx])  # (num_atoms, )

        # if every data points has equal number of atoms
        data = AtomicDataDict.with_batch(data)
        Natoms = bincount(data[AtomicDataDict.BATCH_KEY]) #(batch size)
        if torch.unique(Natoms).numel() == 1: #This works only if all batches have same no of atoms
            natoms = int(Natoms[0])
            # ptr = data["ptr"]
            batch_size = int(torch.unique(data[AtomicDataDict.BATCH_KEY]).shape[0])  #batch_size = ptr.shape[0] - 1
            #This part assumes same no of atoms in all batches
            coeffs = torch.ones((batch_size, natoms + 1, natoms + 1), device = device)
            energy_matrices = torch.stack([
                ewald_bi.get_qeq_matrix(hardness_bi)
                        for (ewald_bi, hardness_bi) in 
                    zip([EwaldAuxiliary(cell=cell_bi, pos=pos_bi, sigmas=sigmas_bi, point_charge=False) for cell_bi, pos_bi, sigmas_bi in 
                        zip(data[AtomicDataDict.CELL_KEY].view(batch_size,3,3), data[AtomicDataDict.POSITIONS_KEY].view(batch_size,-1,3), sigmas.view(batch_size,-1))], 
                        [hardness_bi for hardness_bi in hardness.view(batch_size,-1)] )
                                                                    ])
            coeffs[:, :-1, :-1] = energy_matrices
            coeffs[:, -1, -1] = 0.0
            rhs = torch.cat((-chi.view(batch_size,-1), data[AtomicDataDict.TOTAL_CHARGE_KEY]), dim=-1)
            # Solve Qeq for all batches
            charges_and_lambda = torch.linalg.solve(coeffs, rhs)
            # Extract charges and electrostatic energy
            charges = charges_and_lambda[:, :-1]
            ele = [  0.5 * torch.sum(energy_matrix * charges_bi[:, None] * charges_bi[None, :]) + torch.sum(charges_bi * chi_bi)
                        for energy_matrix, charges_bi, chi_bi in zip(energy_matrices, charges, chi) ]
            ele = [e_qeq_bi / self.scale for e_qeq_bi in ele]
            ele = torch.stack(ele)
            data[AtomicDataDict.CHARGES_KEY] = charges.reshape(batch_size * natoms, -1)

        else:
            ptr = data[AtomicDataDict.BATCH_PTR_KEY]
            batch_size = ptr.shape[0] - 1
            ele = torch.zeros(batch_size, device=device)
            charges = []
            for bi in range(batch_size):
                cell_bi = data[AtomicDataDict.CELL_KEY][bi]
                pos_bi = data[AtomicDataDict.POSITIONS_KEY][ptr[bi] : ptr[bi + 1]]
                sigmas_bi = sigmas[ptr[bi] : ptr[bi + 1]]
                ewald_bi = EwaldAuxiliary(
                    cell=cell_bi,
                    pos=pos_bi,
                    sigmas=sigmas_bi,
                    point_charge=False,
                )

                # coefficient matrix for Qeq
                hardness_bi = hardness[ptr[bi] : ptr[bi + 1]]
                num_atoms_bi = ptr[bi + 1] - ptr[bi]
                coeffs_bi = torch.ones((num_atoms_bi + 1, num_atoms_bi + 1), device=device)
                energy_matrix_bi = ewald_bi.get_qeq_matrix(hardness_bi)
                coeffs_bi[:num_atoms_bi, :num_atoms_bi] = energy_matrix_bi
                coeffs_bi[-1, -1] = 0.0

                total_charge_bi = torch.Tensor([[data[AtomicDataDict.TOTAL_CHARGE_KEY][bi]]]).to(device)  # (1, 1)
                chi_bi = chi[ptr[bi] : ptr[bi + 1]]
                rhs_bi = torch.cat([-chi_bi, total_charge_bi])

                # solve Qeq
                # for small (n, n)-matrix (n < 2048), batched DGESV is faster than usual DGESV in MAGMA
                charges_and_lambda = torch.linalg.solve(
                    torch.unsqueeze(coeffs_bi, dim=0), torch.unsqueeze(rhs_bi, dim=0)
                )
                charges_bi = torch.squeeze(charges_and_lambda, dim=0)[:-1]  # (num_atoms_bi, 1)
                charges.append(charges_bi)

                # minimized electrostatic energy
                e_qeq_bi = 0.5 * torch.sum(
                    energy_matrix_bi * charges_bi[:, None] * charges_bi[None, :]
                )
                e_qeq_bi += torch.sum(charges_bi * chi_bi)
                ele[bi] = e_qeq_bi / self.scale
            data[AtomicDataDict.CHARGES_KEY] = torch.cat(charges) # (num_atoms, 1)

        data[self.out_field] = torch.unsqueeze(ele, dim=1)  # (batch_size, 1)
        return data


class EwaldAuxiliary:
    """
    Parameters
    ----------
    cell: (3, 3)
        cell[i] is the i-th lattice vector
    pos: (num_atoms, 3)
    sigmas: (num_atoms, 1)
        sigmas[i] is the width of the gaussian of the i-th atom
        if point_charge=True, sigmas=None is permitted.
    eta: width of screening gaussian
    cutoff_real: cutoff radius for real part
    cutoff_recip: cutoff radius for reciprocal part
    point_charge: iff true, width of gaussian charges `sigmas` are ignored
    eps: small epsilon to avoid zero division
    """

    # ke = e^{2}/(4 * pi * epsilon_{0}) = 2.3070775523417355e-28 J.m = 14.399645478425668 eV.ang
#    COULOMB_FACTOR: Final[float] = torch.tensor(1e10 * constants.e / (4.0 * math.pi * constants.epsilon_0))

    def __init__(
        self,
        cell: torch.Tensor,
        pos: torch.Tensor,
        sigmas: torch.Tensor, #Optional[torch.Tensor] = None,
        eta: Optional[float] = None,
        cutoff_real: Optional[float] = None,
        cutoff_recip: Optional[float] = None,
        accuracy: float = torch.tensor(1e-8),
        point_charge: bool = True,
        eps: float = 1e-8,
    ):
        self.cell = cell
        self.volume = torch.abs(torch.det(self.cell))          #self.volume = torch.abs(torch.dot(self.cell[0], torch.cross(self.cell[1], self.cell[2])))

        self.pos = pos
        self.sigmas = sigmas

        self.eta = (
            torch.tensor(eta)
            if eta is not None
            else ((self.volume ** 2 / self.pos.shape[0]) ** (1 / 6)) / torch.sqrt(2.0 * torch.tensor(math.pi))
        )
        self.cutoff_real = (
            torch.tensor(cutoff_real) if cutoff_real is not None else torch.sqrt(-2.0 * torch.log(accuracy)) * self.eta
        )
        self.cutoff_recip = (
            torch.tensor(cutoff_recip) if cutoff_recip is not None else torch.sqrt(-2.0 * torch.log(accuracy)) / self.eta
        )

        self.point_charge = point_charge
        self.eps = eps

        # precompute energy matrices
        e_real_matrix = self._calc_real_energy_matrix()
        e_recip_matrix = self._calc_reciprocal_energy_matrix()
        e_self_matrix = self._calc_self_energy_matrix()
        self._e_total_matrix = e_real_matrix + e_recip_matrix + e_self_matrix

    @property
    def num_atoms(self):
        return self.pos.shape[0]

    @property
    def energy_matrix(self):
        """
        total energy matrix e_{ij}
        Ewald summation is obtained by `0.5 * sum_{i,j} e_{ij} q_{i} q_{j}`
        """
        return self._e_total_matrix

    def get_qeq_matrix(self, hardness):
        """
        return energy matrix with hardness term

        Parameters
        ----------
        hardness: (num_atoms, 1)
        """
        assert hardness.shape[0] == self.num_atoms
        mat = torch.clone(self.energy_matrix)
        mat += torch.diag(torch.squeeze(hardness))
        return mat

    def calc_energy(self, charges: torch.Tensor):
        """
        Calculate electrostatic energy by Ewald summation

        Parameters
        ----------
        charges: (num_atoms, 1)
        """
        e_total = 0.5 * torch.sum(self.energy_matrix * charges[:, None] * charges[None, :])
        return e_total


    def _calc_real_energy_matrix(self):
        """
        Calculate real-space-part energy in atomic unit
        """
        # calculate length between atoms `i` and `j` with `shift`
        shifts = get_shifts_within_cutoff(self.cell, self.cutoff_real)  # (num_shifts, 3)
        # disps_ij[i, j, :] is displacement vector r_{ij}
        disps_ij = self.pos[None, :, :] - self.pos[:, None, :]
        disps = disps_ij[None, :, :, :] + torch.matmul(shifts, self.cell)[:, None, None, :]
        distances_all = torch.linalg.norm(disps, dim=-1)  # (num_shifts, num_atoms, num_atoms)

        # retrieve pairs whose length are shorter than cutoff
        within_cutoff = (distances_all > self.eps) & (distances_all < self.cutoff_real)
        distances = distances_all[within_cutoff]

        e_real_matrix_aug = torch.zeros_like(distances_all)
        e_real_matrix_aug[within_cutoff] = torch.erfc(distances / (math.sqrt(2) * self.eta))
        if not self.point_charge:
            gammas_all = torch.sqrt(
                #torch.square(self.sigmas[:, None]) + torch.square(self.sigmas[None, :])
                torch.square(self.sigmas.unsqueeze(1)) + torch.square(self.sigmas.unsqueeze(0))
            )
            gammas = torch.broadcast_to(gammas_all, distances_all.shape)[within_cutoff]
            e_real_matrix_aug[within_cutoff] -= torch.erfc(distances / (math.sqrt(2) * gammas))
        e_real_matrix_aug[within_cutoff] /= distances
        e_real_matrix = 1e10 * constants.e / (4.0 * math.pi * constants.epsilon_0) * torch.sum(
            e_real_matrix_aug, dim=0
        )  # sum over shifts
        return e_real_matrix    

    def _calc_reciprocal_energy_matrix(self):
        # calculate reciprocal points
        recip = get_reciprocal_vectors(self.cell)
        shifts = get_shifts_within_cutoff(recip, self.cutoff_recip)  # (num_shifts, 3)
        ks_all = torch.matmul(shifts, recip)
        length_all = torch.linalg.norm(ks_all, dim=-1)  # (num_shifts, )

        # retrieve reciprocal points whose length are shorter than cutoff
        within_cutoff = (length_all > self.eps) & (length_all < self.cutoff_recip)
        ks = ks_all[within_cutoff]
        length = length_all[within_cutoff]
        # disps_ij[i, j, :] is displacement vector r_{ij}, (num_atoms, num_atoms, 3)
        disps_ij = self.pos[None, :, :] - self.pos[:, None, :]
        phases = torch.sum(ks[:, None, None, :] * disps_ij[None, :, :, :], dim=-1)

        e_recip_matrix_aug = (
            torch.cos(phases)
            * torch.exp(-0.5 * torch.square(self.eta * length[:, None, None]))
            / torch.square(length[:, None, None])
        )
        e_recip_matrix = (
            1e10 * constants.e / (4.0 * math.pi * constants.epsilon_0) #coulomb_factor
            * 4.0
            * math.pi
            / self.volume
            * torch.sum(e_recip_matrix_aug, dim=0)
        )
        return e_recip_matrix

    def _calc_self_energy_matrix(self):
        device = self.pos.device
        diag = -math.sqrt(2.0 / math.pi) / self.eta * torch.ones(self.num_atoms, device=device)
        if not self.point_charge:
            diag += 1.0 / (math.sqrt(math.pi) * self.sigmas.flatten())
        e_self_matrix = 1e10 * constants.e / (4.0 * math.pi * constants.epsilon_0) * torch.diag(diag)
        return e_self_matrix


def get_reciprocal_vectors(cell):
    """
    Return reciprocal vectors of `cell`.
    Let the returned matrix be recip, dot(cell[i, :], recip[j, :]) = 2 * pi * (i == j)
    """
    recip = 2 * math.pi * torch.transpose(torch.linalg.inv(cell), 0, 1)
    return recip


def get_shifts_within_cutoff(cell, cutoff):
    """
    Return all shifts required to search for atoms within cutoff
    """
    device = cell.device

    # projected length for three planes
    proj = torch.zeros(3, device=device)
    nx = torch.cross(cell[1], cell[2])
    ny = torch.cross(cell[2], cell[0])
    nz = torch.cross(cell[0], cell[1])
    proj[0] = torch.dot(cell[0], nx / torch.linalg.norm(nx))
    proj[1] = torch.dot(cell[1], ny / torch.linalg.norm(ny))
    proj[2] = torch.dot(cell[2], nz / torch.linalg.norm(nz))

    shift = torch.ceil(cutoff / torch.abs(proj))
    grid = torch.cartesian_prod(
        torch.arange(float(shift[0]), float(shift[0]) + 1, device=device),
        torch.arange(float(shift[1]), float(shift[1]) + 1, device=device),
        torch.arange(float(shift[2]), float(shift[2]) + 1, device=device),
    )

    return grid
