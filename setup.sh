mkdir -p data
git clone https://github.com/ryancotterell/derviational-paradigms data

python preprocess.py \
  data/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.train.src \
  data/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.train.trg \
  data/train.jsonl \
  data/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.dev.src \
  data/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.dev.trg \
  data/valid.jsonl \
  data/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.test.src \
  data/NOMLEX-plus-training-ADJADV-NOMADV.1.0_v5.test.trg \
  data/test.jsonl \
  data/vocab.json
