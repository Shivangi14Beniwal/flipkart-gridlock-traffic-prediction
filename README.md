# Traffic Demand Prediction
## Flipkart Gridlock Hackathon 2.0

**Score: 90.73 / 100** | 27,600+ participants

## Approach
- Day-48 exact geohash×slot lag as primary feature
- 35+ engineered features (Bayesian encoding, spatial 
  prefix aggregates p4/p5)
- LightGBM (0.75) + CatBoost (0.25) ensemble
- Time-based OOF validation to prevent leakage

## Results
Baseline 88.4 → Final **90.73**

## Tech Stack
Python, LightGBM, CatBoost, Pandas, NumPy, scikit-learn
