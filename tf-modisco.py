# adapted from: https://colab.research.google.com/drive/1-Z6z-CGEAN9zTb0zEaiBOYcQrhiRXI5_#scrollTo=B-pk5Sy-t_m4
import shap
from keras.utils import CustomObjectScope
from keras.metrics import MeanSquaredError
from keras.models import Sequential, Model, load_model
from deeplift.visualization import viz_sequence
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os 
from importlib import reload
import h5py
from collections import OrderedDict
import modisco
import inspect
from collections import Counter
import numpy as np
from modisco.visualization import viz_sequence
reload(viz_sequence)
import modisco.affinitymat.core
reload(modisco.affinitymat.core)
import modisco.cluster.phenograph.core
reload(modisco.cluster.phenograph.core)
import modisco.cluster.phenograph.cluster
reload(modisco.cluster.phenograph.cluster)
import modisco.cluster.core
reload(modisco.cluster.core)
import modisco.aggregator
reload(modisco.aggregator)
import modisco.util
reload(modisco.util)

def LoadConfig(config):
    with open(config, 'r') as file:
        params = json.load(file)
    return params

def one_hot_encode(seq):
    nucleotide_dict = {'A': [1, 0, 0, 0],
                       'C': [0, 1, 0, 0],
                       'G': [0, 0, 1, 0],
                       'T': [0, 0, 0, 1],
                       'N': [0, 0, 0, 0]} 
    return np.array([nucleotide_dict[nuc] for nuc in seq])

def kmer_encode(sequence, k=3):
    sequence = sequence.upper()
    kmers = [sequence[i:i+k] for i in range(len(sequence) - k + 1)]
    kmer_counts = Counter(kmers)
    return {kmer: kmer_counts.get(kmer, 0) / len(kmers) for kmer in [''.join(p) for p in product('ACGT', repeat=k)]}

def kmer_features(seq, k=3):
    all_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    feature_matrix = []
    kmer_freqs = kmer_encode(seq, k)
    feature_vector = [kmer_freqs[kmer] for kmer in all_kmers]
    feature_matrix.append(feature_vector)
    return np.array(feature_matrix)

def prepare_input(data_set, params):
    if params['encode'] == 'one-hot':
        seq_matrix = np.array(data_set['Sequence'].apply(one_hot_encode).tolist())  # (number of sequences, length of sequences, nucleotides)
    elif params['encode'] == 'k-mer':
        seq_matrix = np.array(data_set['Sequence'].apply(kmer_features, k=3).tolist())  # (number of sequences, 1, 4^k)
    else:
        raise Exception ('wrong encoding method')

    Y_dev = data_set.Dev_log2_enrichment
    Y_hk = data_set.Hk_log2_enrichment
    Y = [Y_dev, Y_hk]

    return seq_matrix, Y

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def modisco_motif_plots(task):
    if task == 'Dev_contrib_scores': hdf5_results = h5py.File("/content/drive/MyDrive/DeepSTARR_tutorial/Dev_modisco_results.hdf5","r") # from Google Drive
    if task == 'Hk_contrib_scores': hdf5_results = h5py.File("/content/drive/MyDrive/DeepSTARR_tutorial/Hk_modisco_results.hdf5","r") # from Google Drive

    metacluster_names = [
        x.decode("utf-8") for x in 
        list(hdf5_results["metaclustering_results"]["all_metacluster_names"][:])]

    all_patterns = []


    if task == 'Dev_contrib_scores': background = np.mean(X_cand_dev, axis=(0,1))
    if task == 'Hk_contrib_scores': background = np.mean(X_cand_hk, axis=(0,1))

    # sequence background
    if task == 'Dev_contrib_scores': background = np.mean(X_cand_dev, axis=(0,1))
    if task == 'Hk_contrib_scores': background = np.mean(X_cand_hk, axis=(0,1))

    for metacluster_name in metacluster_names:
        print(metacluster_name)
        metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
                                       [metacluster_name])
        print("activity pattern:",metacluster_grp["activity_pattern"][:])
        all_pattern_names = [x.decode("utf-8") for x in 
                             list(metacluster_grp["seqlets_to_patterns_result"]
                                                 ["patterns"]["all_pattern_names"][:])]
        if (len(all_pattern_names)==0):
            print("No motifs found for this activity pattern")
        for pattern_name in all_pattern_names:
            print(metacluster_name, pattern_name)
            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
            # print("Actual importance scores:")
            # viz_sequence.plot_weights(pattern[str(task + "_contrib_scores")]["fwd"])
            # print("Hypothetical scores:")
            # viz_sequence.plot_weights(pattern[str(task + "_hypothetical_contribs")]["fwd"])
            print("IC-scaled, fwd and rev:")
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                            background=background)) 
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                            background=background)) 

            #Plot the subclustering too, if available
            if ("subclusters" in pattern):
                print("PLOTTING SUBCLUSTERS")
                subclusters = np.array(pattern["subclusters"])
                twod_embedding = np.array(pattern["twod_embedding"])
                plt.scatter(twod_embedding[:,0], twod_embedding[:,1], c=subclusters, cmap="tab20")
                plt.show()
                for subcluster_name in list(pattern["subcluster_to_subpattern"]["subcluster_names"]):
                    subpattern = pattern["subcluster_to_subpattern"][subcluster_name]
                    print(subcluster_name.decode("utf-8"), "size", len(subpattern["seqlets_and_alnmts"]["seqlets"]))
                    subcluster = int(subcluster_name.decode("utf-8").split("_")[1])
                    plt.scatter(twod_embedding[:,0], twod_embedding[:,1], c=(subclusters==subcluster))
                    plt.show()
                    # viz_sequence.plot_weights(subpattern[str(task + "_hypothetical_contribs")]["fwd"])
                    # viz_sequence.plot_weights(subpattern[str(task + "_contrib_scores")]["fwd"])
                    viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(subpattern["sequence"]["fwd"]),
                                                            background=background))

    hdf5_results.close()

def main(config, indir, mdldir, outdir):
    data = pd.read_table(indir)
    main_params = LoadConfig(config)
    X_test, Y_test = prepare_input(data[data['set'] == "Test"], main_params)
    with CustomObjectScope({'mse': MeanSquaredError}):
        main_model = load_model(mdldir)

    # Prepare DeepExplainer for developmental and housekeeping model output
    np.random.seed(seed=1234)
    background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # this is required due to conflict between versions (https://github.com/slundberg/shap/issues/1110)
    explainer_dev = shap.DeepExplainer((main_model.layers[0].input, main_model.layers[-2].output),
                                    data=background)
    explainer_hk = shap.DeepExplainer((main_model.layers[0].input, main_model.layers[-1].output),
                                    data=background)
    cand_dev = data.loc[(data['set'] == 'Test') &
                        (data['ID'].str.contains('_\+_positive_peaks')) & 
                        (data['Dev_log2_enrichment'] > 3)]
    cand_dev = cand_dev.sort_values('Dev_log2_enrichment', ascending = False)
    X_cand_dev, Y_cand_dev = prepare_input(cand_dev, main_params)
    pred_values_dev = main_model.predict(X_cand_dev)[0].squeeze() # 0 output for developmental
    shap_values_dev = explainer_dev.shap_values(X_cand_dev)[0]
    shap_values_dev = np.expand_dims(shap_values_dev.squeeze(-1), axis=0).repeat(619, axis=0)
    final_contr_scores_dev = shap_values_dev*X_cand_dev

    cand_hk = data.loc[(data['set'] == 'Test') &
                        (data['ID'].str.contains('_\+_positive_peaks')) & 
                        (data['Hk_log2_enrichment'] > 3)]

    cand_hk = cand_hk.sort_values('Hk_log2_enrichment', ascending = False)
    X_cand_hk, Y_cand_hk = prepare_input(cand_hk, main_params)
    pred_values_hk = main_model.predict(X_cand_hk)[1].squeeze() # 0 output for housekeeping
    shap_values_hk = explainer_hk.shap_values(X_cand_hk)[0]
    shap_values_hk = np.expand_dims(shap_values_hk.squeeze(-1), axis=0).repeat(539, axis=0)
    final_contr_scores_hk = shap_values_hk*X_cand_hk

    f = h5py.File(outdir+'contr_scores.h5', 'w')
    g = f.create_group("contrib_scores")
    g.create_dataset('Dev_contrib_scores', data=final_contr_scores_dev)
    g.create_dataset('Hk_contrib_scores', data=final_contr_scores_hk)

    g = f.create_group("hyp_contrib_scores")
    g.create_dataset('Dev_contrib_scores', data=shap_values_dev)
    g.create_dataset('Hk_contrib_scores', data=shap_values_hk)

    tasks = f["contrib_scores"].keys()
    task_to_scores = OrderedDict()
    task_to_hyp_scores = OrderedDict()
    for task in tasks:
        task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task]]
        task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task]]
    f.close()

    get_default_args(modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow)
    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)

    dev_tfmodisco_results = my_tfmodisco('Dev_contrib_scores')
    grp = h5py.File("/content/drive/MyDrive/DeepSTARR_tutorial/Dev_modisco_results.hdf5", "w") # on Google Drive
    dev_tfmodisco_results.save_hdf5(grp)
    grp.close()
    modisco_motif_plots('Dev_contrib_scores')

    hk_tfmodisco_results = my_tfmodisco('Hk_contrib_scores')
    grp = h5py.File("/content/drive/MyDrive/DeepSTARR_tutorial/Hk_modisco_results.hdf5", "w") # on Google Drive
    hk_tfmodisco_results.save_hdf5(grp)
    grp.close()
    modisco_motif_plots('Hk_contrib_scores')

    if task == 'Dev_contrib_scores': one_hot = X_cand_dev
    if task == 'Hk_contrib_scores': one_hot = X_cand_hk

    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
                        target_seqlet_fdr=0.2,
                        sliding_window_size=15,
                        flank_size=5,
                        max_seqlets_per_metacluster=50000,
                        seqlets_to_patterns_factory=
                            modisco.tfmodisco_workflow
                                    .seqlets_to_patterns
                                    .TfModiscoSeqletsToPatternsFactory(
                                trim_to_window_size=15,
                                initial_flank_to_add=5,
                                final_min_cluster_size=30
                        )
                   )(
                task_names=[task],
                contrib_scores=task_to_scores,
                hypothetical_contribs=task_to_hyp_scores,
                one_hot=one_hot,
                null_per_pos_scores = null_per_pos_scores)

if (len(sys.argv)!=5):
    exit("<parms.config> <mdl-dir> <data-indir> <outdir> \n")

(config, indir, mdldir, outdir)=sys.argv[1:]
main(config, indir, mdldir, outdir)