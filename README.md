## Examine

This is an experiment with a synthetically generated author biographies (bio description).
Model : biodescription on X/Twitter \\
Model : Claude Sonnet 5.5 (medium effort) \\ 
Date : 15th October 2026 \\

---

### Prompt 1 (for data generation)

```bash

Create dataset for NLP experiments. 

Could you create a synthetic data set of around 300 rows.
I'm imitating twitter/X user bio descriptions to test the best embedding models, in a csv format with different versions of this data. The different ideas are:

- with flags only : but contrasting such as israel palestine, india pakistan, ireland england
flags + some text
- with very short sentences (3 tokens max) with contrasting ideas in #India #Pakistan #vaccinationSavesLives #AntiVax etc #prolife, #prochoice
- some regular mixed sentences of types above of longer length (max 10 /15)
- some other typical authorbioDescription kind of sentences related to other topis such as profession, health, environment, sports , philosophy etc.
- a mix and match of these combinations.
- contrasting ideas should represent different biographies mostly, you can have 5 or 6 mixed constrasting ones.
```


### Prompt 2
```bash 

Could you add another 200 rows. 
Change the columns ids  to bio_id and bio_description. 

Add some more bios with emoticons of emotions smileys etc. reiligious emojis, some food emojis, some just with emojis and some others mixed with texts. 
Avoid adding | . 
```

---


The experiment has one pipeline:

1. Embed each configured text version and model.
2. Cluster every embedding file with the configured runs.
3. Build cluster tables and 2D visualizations.

Run it from the repository root with:

```bash
uv run examine
```

Use a different configuration or recompute existing embeddings with:

```bash
uv run examine --config experiment.yaml --overwrite
```

The configured data root contains `processed/`, `embeddings/`, `clusters/`, and
`viz/` directories. The canonical implementation lives in `src/examine`.
