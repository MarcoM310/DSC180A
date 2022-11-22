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
&nbsp;&nbsp;&nbsp;&nbsp;    ├── data           <- Scripts to download or generate data.<br>
&nbsp;&nbsp;&nbsp;&nbsp;    │   └── make_dataset.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;    ├── features       <- Scripts to turn raw data into features for modeling.<br>
&nbsp;&nbsp;&nbsp;&nbsp;    │   └── build_features.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;    ├── models         <- Scripts to train models and make predictions.<br>
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── predict_model.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;    │   └── train_model.py<br>
&nbsp;&nbsp;&nbsp;&nbsp;    └── visualization  <- Scripts to create exploratory and results-oriented viz.<br>
&nbsp;&nbsp;&nbsp;&nbsp;        └── visualize.py<br>
