# ai4privacy multilingual slice (test fixture)

3,928 texts from [ai4privacy/open-pii-masking-500k-ai4privacy](https://huggingface.co/datasets/ai4privacy/open-pii-masking-500k-ai4privacy):
2,592 English, 747 Hindi and 589 Telugu, used by `tests/test_multilingual_ai4privacy.py`.

| File | Fields | Read by |
|---|---|---|
| `inputs.jsonl` | `uid` (the dataset's row uid), `lang`, `text` | the pipeline, via `decisions log` (what shadow mode would capture) |
| `judge_key.jsonl` | `sha256` of the stripped text, `label` | only judges: `tests/hardcoded_judge.py` and the mock judge (`src/system_none/mock_judge.py`) |

The two files share no field except through the text hash, so a label can reach the student only
by being asked of the judge.

**Labels.** Each text gets one label from the mask types in its `privacy_mask`, taking the first
group that matches: `id_number` (IDCARDNUM, PASSPORTNUM, DRIVERLICENSENUM, SOCIALNUM, TAXNUM,
CREDITCARDNUMBER, …), then `contact` (EMAIL, TELEPHONENUM), then `address` (STREET, BUILDINGNUM,
CITY, ZIPCODE), otherwise `identity_only` (names, age, sex/gender, title, dates and times only).

**Selection.** English, Hindi and Telugu rows from the first 12,000 rows of the `validation` split,
read in pages through the Hugging Face datasets-server: every such row from row 3,200 on, and a
sample before that. Removed: 4 empty texts, rows whose script does not match their stated language
(some rows tagged `hi` are in other Indic scripts), and repeated texts (the text "..." appears 21
times and one Hindi text twice; one copy of each is kept). Rows are in `uid` order.

| | id_number | contact | address | identity_only |
|---|---|---|---|---|
| English | 455 | 567 | 532 | 1,038 |
| Hindi | 128 | 168 | 131 | 320 |
| Telugu | 153 | 128 | 102 | 206 |

**License.** Ai4Privacy publishes the dataset under CC BY 4.0, and because it was generated with
Llama 3.1 and 3.3, use and redistribution also follow the Llama Community License. This directory
therefore carries a `NOTICE` ("Built with Llama" and the required attribution lines) and copies of
both licenses in `licenses/`, taken unchanged from the dataset repository. The texts are
synthetic: names, numbers and addresses in them are generated, not real people's data. The slice
is unchanged apart from the selection and the derived label.

The tests train throwaway models on this slice and never publish them. If you train a model on
this data and distribute it, the Llama license also requires its name to start with "Llama".
