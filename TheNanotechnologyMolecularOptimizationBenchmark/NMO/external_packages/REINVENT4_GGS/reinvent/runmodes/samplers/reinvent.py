"""The Reinvent sampling module"""

__all__ = ["ReinventSampler"]
import logging

import numpy as np
from rdkit import Chem

from .sampler import Sampler, remove_duplicate_sequences, validate_smiles
from .params import SAMPLE_BATCH_SIZE
from ...models.model_factory.sample_batch import SampleBatch, SmilesState
from reinvent.models.reinvent.models.vocabulary import IndexTokenizer

logger = logging.getLogger(__name__)


class ReinventSampler(Sampler):
    """Carry out sampling with Reinvent"""

    def sample(self, dummy) -> SampleBatch:
        """Samples the Reinvent model for the given number of SMILES

        :param dummy: Reinvent does not need SMILES input
        :returns: SampleBatch
        """

        batch_sizes = [SAMPLE_BATCH_SIZE for _ in range(self.batch_size // SAMPLE_BATCH_SIZE)]

        if remainder := self.batch_size % SAMPLE_BATCH_SIZE:
            batch_sizes += [remainder]

        sequences = []

        for batch_size in batch_sizes:
            for batch_row in self.model.sample(batch_size):
                sequences.append(batch_row)

        sampled = SampleBatch.from_list(sequences)
        sampled.items1 = None

        if self.unique_sequences:
            sampled = remove_duplicate_sequences(sampled, is_reinvent=True)

        if isinstance(getattr(self.model, "tokenizer", None), IndexTokenizer):
            # Index-encoded sequences: bypass RDKit, validate as space-separated digit strings
            sampled.smilies = list(sampled.output)
            sampled.states = np.array([
                SmilesState.VALID
                if (seq and seq.strip() and all(t.isdigit() for t in seq.strip().split()))
                else SmilesState.INVALID
                for seq in sampled.output
            ])
        else:
            mols = [
                Chem.MolFromSmiles(smiles, sanitize=False) if smiles else None
                for smiles in sampled.output
            ]
            sampled.smilies, sampled.states = validate_smiles(mols, sampled.output)

        return sampled
