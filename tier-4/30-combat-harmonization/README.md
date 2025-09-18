## Project 30: ComBat Harmonization and Generalization Study

### Overview
Quantify site effects, apply ComBat harmonization, and test generalization performance.

### Instructions
Provide CSV with features, `site` column, and labels. Run:
```bash
python combat_cli.py --features feats.csv --site_col site --label_col y --out_dir outputs/combat
```

### Learning: Site Effects
ComBat removes batch/site effects via empirical Bayes. Evaluate pre/post harmonization performance with leave-site-out CV.

