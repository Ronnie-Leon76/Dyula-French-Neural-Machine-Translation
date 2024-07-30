"""Koumankan_mt_dyu_fr_tfds dataset."""

import tensorflow_datasets as tfds
import pandas as pd
from pathlib import Path


class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Koumankan_mt_dyu_fr_tfds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=("This dataset contains translations from Dyula to French."),
        features=tfds.features.FeaturesDict({
            'dyu': tfds.features.Text(),
            'fr': tfds.features.Text(),
        }),
        supervised_keys=('dyu', 'fr'),
        homepage='https://huggingface.co/datasets/uvci/Koumankan_mt_dyu_fr',
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    extracted_path = {
        'train': Path('./data/train_refined.csv'),
        'validation': Path('./data/val_refined.csv'),
        'test': Path('./data/test_refined.csv'),
    }

    return {
        tfds.Split.TRAIN: self._generate_examples(extracted_path['train']),
        tfds.Split.VALIDATION: self._generate_examples(extracted_path['validation']),
        tfds.Split.TEST: self._generate_examples(extracted_path['test']),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    data = pd.read_csv(path)
    for idx, row in data.iterrows():
      yield idx, {
          'dyu': row['dyu'],
          'fr': row['fr'],
      }
