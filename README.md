Hello! This is the repo for running the Substate-Handling Iterative Vector Assessment for amide hydrogen chemical shifts - a.k.a. SHIVA.

This program is still a work in progress, but the basics are there. This should allow you to generate chemical shift estimates for the amide hydrogens of any protein for which you have an ensemble of structures.

There are two supported ways to run an ensemble right now. Either you can collect a series of PDB files and put them in a directory together or you can run BioEmu to generate a trajectory xtc file. I recommend the latter, since it's what the program was trained with. Additionally, if you do use your own PDBs, I recommend removing hydrogens. The program will still run if you have your own hydrogens, but it has functionality for adding them, and hydrogen nomenclature / placement can be a big source of error.

Here are the versions of my major dependencies:
python		3.12.9
bioemu		0.1.12
numpy		2.2.5
scipy		1.15.2
pandas		2.2.3
openmm		8.2
MDAnalysis	2.9.0
mdtraj		1.10.3
xgboost		3.0.0

BioEmu is not a strict dependency of this repo - it's just what I recommend installing if you want good ensembles as input. It's a pretty easy install on most machines, though it does require cuda>=12 to use the md version (which you do need to have installed). Refer to their github for more info:
https://github.com/microsoft/bioemu

xgboost is never actually imported but it needs to be there for the unpickling of the models to work.

MDAnalysis can be avoided if you use a PDB ensemble, rather than the BioEmu one (though again, I do recommend BioEmu). I might write a workaround later to avoid the MDAnalysis install.

To start you need to generate features. Go to the home directory. If you are using BioEmu, run this:
python scripts/run_pipeline_batch_individualDSSP_hbond_withSS_predictOnly.py --input-dir input_xtc --output-dir feats

If you are using a directory with PDB files, run this instead:
python scripts/run_pipeline_batch_individualDSSP_hbond_withSS_predictOnly_pdbDir.py --input-dir input_pdb --output-dir feats

Following this, the scripts you need to run are the same, and are as follows:
python scripts/add_ensemble_statistics_to_features_v3.py feats feats_SS_ensemble
python scripts/split_ss_types_v2.py feats_SS_ensemble feats_SS_ensemble_H feats_SS_ensemble_E feats_SS_ensemble_C
python scripts/xgboost_loadedModel_predOnly_C.py
python scripts/xgboost_loadedModel_predOnly_E.py
python scripts/xgboost_loadedModel_predOnly_H.py
python scripts/append_HEC_results.py
python scripts/xgboost_aggregate_and_eval.py test_predictions_all.csv

After this, your predictions should be in test_predictions_all.csv

Note that the prediction will run for ALL subfolders containing data in the directory specified by --input-dir

I've included here one example, which is the protein associated with BMRB entry 6395. There's a prediction based on the BioEmu ensemble, and another based on the NMR ensemble. Reassuringly, the former is more accurate! (Though I'll still need to do more testing - I think this one wasn't a super accurate NMR ensemble.)

Happy predicting!
