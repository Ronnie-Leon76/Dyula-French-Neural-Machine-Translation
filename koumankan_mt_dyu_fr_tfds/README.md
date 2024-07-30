# Koumankan_mt_dyu_fr_tfds Dataset

This dataset is a collection of translations from Dyula to French, aimed at supporting research in machine translation, natural language processing, and linguistics. The dataset is structured to provide a straightforward way for models to learn Dyula to French translation.

## Dataset Description

- **Homepage:** [Dataset Homepage](https://huggingface.co/datasets/uvci/Koumankan_mt_dyu_fr)
- **Point of Contact:** [Contact Email](mailto:ronleon76@gmail.com)

### Download

To download the dataset, use TensorFlow Datasets (TFDS). Here is an example snippet to load the dataset in TensorFlow:

```python
import tensorflow_datasets as tfds

ds, info = tfds.load('koumankan_mt_dyu_fr_tfds', with_info=True)
```