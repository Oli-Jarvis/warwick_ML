from typing import Dict, List, Optional
import logging
import os

from rdkit.Chem.rdmolfiles import MolToSmiles

from reinvent.chemistry import conversions
from reinvent.chemistry.standardization.filter_configuration import FilterConfiguration
from reinvent.chemistry.standardization.filter_registry import FilterRegistry
from reinvent.action_space import Action_Space_GroupSelfies  # ← add this

logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')


class RDKitStandardizer:
    def __init__(
        self, filter_configs: Optional[List[FilterConfiguration]], isomeric=False, *args, **kwargs
    ):
        self._filter_configs = self._set_filter_configs(filter_configs)
        self._filters = self._load_filters(self._filter_configs)
        self.isomeric = isomeric

        for config in self._filter_configs:
            logger.info(f"Applying filter {config.name} to input SMILES")

        if self.isomeric:
            logger.info("Stereochemistry kept in input SMILES")

        self._action_space = None

        self._skip_filters = os.environ.get("REINVENT_SKIP_FILTERS", "0") == "1"
        if self._skip_filters:
            logger.info("REINVENT_SKIP_FILTERS=1: RDKit filters disabled for index sequences")


    def _get_action_space(self):
        """Lazy-load action space — only used when encoding_type is index."""
        if self._action_space is None:
            # grammar_path stored in comments of model metadata
            # for now read from environment variable or hardcode path
            import os
            grammar_path = os.environ.get("REINVENT_GRAMMAR_PATH")
            if grammar_path is None:
                raise RuntimeError(
                    "Set REINVENT_GRAMMAR_PATH env variable to your grammar file "
                    "when using index encoding"
                )
            self._action_space = Action_Space_GroupSelfies.from_grammar_path(grammar_path)
        return self._action_space

    def _is_index_sequence(self, smile: str) -> bool:
        """Check if string is a space-separated index sequence rather than SMILES."""
        tokens = smile.strip().split()
        return len(tokens) > 1 and all(t.isdigit() for t in tokens)

    def _index_to_smiles(self, index_seq: str) -> str | None:
        """Convert index sequence → Group SELFIES → SMILES. Returns None if invalid."""
        try:
            action_space = self._get_action_space()
            action_sequence = [int(x) for x in index_seq.strip().split()]
            #print(action_sequence)

            if not action_space.has_end_token(action_sequence):
                #print("no end token")
                return None

            group_selfies = action_space.action_sequence_to_encoding(action_sequence)
            #print(f"group_selfies: {group_selfies}")
            if not group_selfies:
                #print("no group_selfies")
                return None

            # Group SELFIES → SMILES via the grammar decoder
            mol = action_space.selfies_grammar.decoder(group_selfies)
            smiles = MolToSmiles(mol)
            #print(f"smiles {smiles=}")
            return smiles

        except Exception as e:
            print(e)
            return None

    def apply_filter(self, smile: str) -> str:
        # --- INDEX ENCODING HANDLING ---
        if self._is_index_sequence(smile):
            smiles_for_validation = self._index_to_smiles(smile)
            if smiles_for_validation is None:
                message = f'"default" filter: {smile} is invalid'
                logger.warning(message)
                return None

            if self._skip_filters:
                return smile

            # Run RDKit filters on the SMILES for proper validation
            molecule = conversions.smile_to_mol(smiles_for_validation)
            for config in self._filter_configs:
                if molecule:
                    rdkit_filter = self._filters[config.name]
                    print(f"DEBUG: applying filter '{config.name}' to {MolToSmiles(molecule)}")
                    if config.parameters:
                        molecule = rdkit_filter(molecule, **config.parameters)
                    else:
                        molecule = rdkit_filter(molecule)
                    if molecule:
                        print(f"DEBUG: '{config.name}' → PASSED")
                    else:
                        print(f"DEBUG: '{config.name}' → KILLED molecule")
                else:
                    print(f"DEBUG: '{config.name}' → skipped (molecule already None)")
                    message = f'"{config.name}" filter: {smile} is invalid'
                    logger.warning(message)
            if not molecule:
                message = f'"{self._filter_configs[-1].name}" filter: {smile} is invalid'
                logger.warning(message)
                return None
            return smile

        # --- ORIGINAL SMILES HANDLING (unchanged) ---
        molecule = conversions.smile_to_mol(smile)
        for config in self._filter_configs:
            if molecule:
                rdkit_filter = self._filters[config.name]
                if config.parameters:
                    molecule = rdkit_filter(molecule, **config.parameters)
                else:
                    molecule = rdkit_filter(molecule)
            else:
                message = f'"{config.name}" filter: {smile} is invalid'
                logger.warning(message)
        if not molecule:
            message = f'"{self._filter_configs[-1].name}" filter: {smile} is invalid'
            logger.warning(message)
            return None
        valid_smile = MolToSmiles(molecule, isomericSmiles=self.isomeric)
        return valid_smile

    def _set_filter_configs(self, filter_configs):
        return (
            filter_configs
            if filter_configs
            else [FilterConfiguration(name="default", parameters={})]
        )

    def _load_filters(self, filter_configs: List[FilterConfiguration]) -> Dict:
        registry = FilterRegistry()

        return {
            filter_config.name: registry.get_filter(filter_config.name)
            for filter_config in filter_configs
        }
