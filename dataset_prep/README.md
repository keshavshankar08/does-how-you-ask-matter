# Data

The prepared files are in `data/`. Python 3.10+ is enough to load them.

```python
from prepare_data import load_jsonl

medquad = list(load_jsonl("data/medquad.jsonl"))
amqa = list(load_jsonl("data/amqa.jsonl"))
```

Both use `question` and `reference_answer`. AMQA also has `options` and `answer_idx`. The original text is unchanged.

- `medquad.jsonl` has 16,375 questions. Of the 47,457 source records, 31,034 have no answer and 48 are duplicate QA pairs.
- `amqa.jsonl` has all 801 base questions. Each answer letter was checked against its option text.
- `excluded.jsonl` keeps the removed records and the reason for each removal.
- `amqa_variants.jsonl` has the 4,806 released variants, linked by `record_id`. Some add clinical clues, so they need checking before use as framing-only changes.
- `summary.json` has the counts, source versions and checksums.

`candidate` means a record passed the missing-text and duplicate checks, not that its medical content has been reviewed. `review_flags` mark possible issues; differing reference answers are kept. Keep each `question_group_id` together when splitting data.

## Rebuild and test

Run these from this directory. Rebuilding also needs Git and a new output directory.

```sh
git clone https://github.com/abachaa/MedQuAD.git sources/MedQuAD
git -C sources/MedQuAD checkout 577bd37b96c02d1833b2c9eed2de9f96964e96cb
git clone https://github.com/XY-Showing/AMQA.git sources/AMQA
git -C sources/AMQA checkout 335b1752ba4912d00050fa6531ba8d2c2aef6fb9
python prepare_data.py --sources sources --output data_rebuilt
python -m unittest -v test_prepare_data.py
```

The script leaves existing output directories alone. Raw sources and rebuilt copies are in `.gitignore`.

[MedQuAD](https://github.com/abachaa/MedQuAD) is by Asma Ben Abacha and Dina Demner-Fushman, under CC BY 4.0; the license is in `licenses/`. [AMQA / FairMedQA](https://github.com/XY-Showing/AMQA) is by Ying Xiao and colleagues ([paper](https://arxiv.org/abs/2505.19562)), with CC BY 4.0 listed on the authors' [dataset card](https://huggingface.co/datasets/Showing-KCL/AMQA). Changes here are parsing, screening, deduplication and metadata.
