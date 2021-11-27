# Image Saliency Map Experiments

Please follow the steps below to run the experiments. They also introduce the purpose of each file.

0. The custom dataset of confusing bird species is in the folder `bird_dataset`, with sub-folders named `train`, `val` and `test`. The folder structure follow that of the ImageNet.
1. The five manipulation types (plus the no-op) are defined in `manipulations.py`. Each is a sub-class of `Manipulation` object. Please consult the documentation within for more information. The `test_manipulation()` function provides a usage demo.
2. The dataset modification procedure is defined in `dataset_modification.py`, and `DatasetModifier` specifically. Please consult the documentation within for parameters. The `main()` function provides a usage demo.
3. For each experiment, the correspondingly modified datasets need to be created first. Note that one modified dataset needs to be created for every run, and the modified dataset size is around 650MB. Overall, the dataset and the trained models for all experiments could total around 500GB.
    * For the first and last experiment, run `python dm_attr_vs_er.py`, which will create a folder named `attr_vs_er_experiment` with the datasets for 100 runs.
    * For the second experiment, run `python dm_visibility.py`, which will create a folder named `visibility_experiment` with the datasets for 100 runs.
    * For the third experiment, run `python dm_orig_corr_strength.py`, which will create a folder named `orig_corr_strength_experiment` with the datasets for 120 runs.
4. To train the model for each experiment, uncomment the corresponding line in `__main__` of `train.py`, and run the file. The models will be saved to a newly created `training` folder under each modified dataset folder.
5. All the saliency maps are implemented in `saliency.py`. Please consult the documentation within for more information.
