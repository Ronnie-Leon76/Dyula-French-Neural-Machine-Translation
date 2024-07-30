"""Koumankan_mt_dyu_fr_tfds dataset."""

import koumankan_mt_dyu_fr_tfds_dataset_builder
import tensorflow_datasets as tfds

class KoumankanMtDyuFrTfdsTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for Koumankan_mt_dyu_fr_tfds dataset."""
  DATASET_CLASS = koumankan_mt_dyu_fr_tfds_dataset_builder.Builder
  SPLITS = {
      'train': 8065,
      'validation': 1471,
      'test': 1393,
  }
  OVERLAPPING_SPLITS = {
      'train': ['validation'],
      'validation': ['test'],
  }


if __name__ == '__main__':
  tfds.testing.test_main()
