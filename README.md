# CDTR

CDTR transfers a character detector to historical Chinese text recognition. The model reads a cropped vertical column and returns its transcription. Source training uses character labels and boxes. Target adaptation copies classifier rows by character identity and trains on column transcriptions with CTC. At inference, the decoder can use class priors estimated from the target training images.

## Repository layout

| Path | Contents |
| --- | --- |
| [`main_synthetic.py`](main_synthetic.py) | Character detection pretraining |
| [`finetuning.py`](finetuning.py) | Character mapping and CTC adaptation |
| [`models/`](models/) | Recognition model and deformable attention extension |
| [`config/`](config/) | Model and dataset configurations |
| [`tools/`](tools/) | Dataset preparation and evaluation |
| [`util/prior_decoding.py`](util/prior_decoding.py) | Character-prior correction and CTC collapse |

## Installation

Use Linux with a CUDA toolkit and a matching PyTorch and torchvision installation. Install the remaining dependencies and build the deformable attention extension from the repository root.

```bash
python -m pip install -r requirements.txt
(cd models/dino/ops && bash make.sh)
```

## Data

Obtain [MTHv2](https://github.com/HCIILAB/MTHv2_Datasets_Release) and [HDRC](https://doi.org/10.1109/ICDAR.2019.00241) from their providers. The dataset loaders read prepared columns from `data/` by default. Edit [`datasets/config.json`](datasets/config.json) to change the data root. Each prepared dataset uses a `lines/` directory and a `labels.pkl` file containing `train`, `valid`, and `test` splits. Source character detection also uses the original box annotations.

```bash
python tools/prepare_tkhmth2200_single_column.py --src-root data/TKHMTH2200 --dst-root data
python tools/prepare_hdrc_dtlr.py --image-root data/HDRC --xml-root data/HDRC_GT --output-root data/hdrc_dtlr
```

The preparation commands use the tools' default splitting and crop options. For a controlled comparison, keep the same prepared split and character vocabulary across models. Use your own copies of the datasets and checkpoints.

## Training and evaluation

Replace the checkpoint and character vocabulary paths with your own assets. `source_charset.pkl` must list the source characters in classifier row order.

```bash
# Train a character detector on the source columns.
python main_synthetic.py -c config/MTHV2_stage1_query_budget.py \
  --dataset_file mth_combo --output_dir outputs/source

# Adapt the source detector using 5% of the target training columns.
python finetuning.py -c config/SAQT_HDRC_low_resource.py \
  --dataset_file mth1000 --output_dir outputs/hdrc5 \
  --pretrain_model_path path/to/source_checkpoint.pth \
  --new_class_embedding --smart_mapping --path_old_charset path/to/source_charset.pkl \
  --options mth1000_train_fraction=0.05

# Evaluate the checkpoint selected on validation with greedy CTC.
python tools/eval_ctc_micro_batch.py -c config/SAQT_HDRC.py \
  --dataset_file mth1000 --checkpoint path/to/selected_checkpoint.pth \
  --split test --batch_size 1 --new_class_embedding --output_json outputs/hdrc5/test.json
```

`config/SAQT_MTHV2.py` configures MTHv2 recognition. The `SAQT_*` and `*_dtlr.py` names are retained for compatibility with existing configurations and checkpoints. The evaluation script reports character error rate (CER), accuracy rate (AR), and correct rate (CR), aggregated over reference characters.

### Character-prior decoding

Sum the nonblank query probabilities over the target **training** images, in the target vocabulary order. Pass those sums and the total number of training characters to `log_character_prior`. On validation data, choose the prior strength and nonblank offset, then reuse them for test. `decode_with_character_prior` takes vertically ordered query probabilities with blank at index zero and returns zero-based character indices.

## License and acknowledgments

The source is distributed under [Apache 2.0](LICENSE). The implementation builds on [DTLR](https://github.com/raphael-baena/DTLR), [DINO](https://github.com/IDEA-Research/DINO), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR). Character-prior correction follows the model-prior idea in [Prior2Posterior](https://openaccess.thecvf.com/content/WACV2025/html/Bhat_Prior2Posterior_Model_Prior_Correction_for_Long-Tailed_Learning_WACV_2025_paper.html).
