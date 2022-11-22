# DSC180A

├── .gitignore         <- Files to keep out of version control (e.g. data/binaries).<br>
├── run.py             <- run.py with calls to functions in src.<br>
├── README.md          <- The top-level README for developers using this project.<br>
├── data<br>
│   ├── temp           <- Intermediate data that has been transformed.<br>
│   ├── out            <- The final, canonical data sets for modeling.<br>
│   └── raw            <- The original, immutable data dump.<br>
├── notebooks          <- Jupyter notebooks (presentation only).<br>
├── references         <- Data dictionaries, explanatory materials.<br>
├── requirements.txt   <- For reproducing the analysis environment, e.g.<br>
│                         generated with `pip freeze > requirements.txt`<br>
├── src                <- Source code for use in this project.<br>
    ├── data           <- Scripts to download or generate data.<br>
    │   └── make_dataset.py<br>
    ├── features       <- Scripts to turn raw data into features for modeling.<br>
    │   └── build_features.py<br>
    ├── models         <- Scripts to train models and make predictions.<br>
    │   ├── predict_model.py<br>
    │   └── train_model.py<br>
    └── visualization  <- Scripts to create exploratory and results-oriented viz.<br>
        └── visualize.py<br>
