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

def prepare_input(data_set):
    if main_params['encode'] == 'one-hot':
        seq_matrix = np.array(data_set['Sequence'].apply(one_hot_encode).tolist())  # (number of sequences, length of sequences, nucleotides)
    elif main_params['encode'] == 'k-mer':
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
    if task == 'Dev_contrib_scores': hdf5_results = h5py.File(outdir+"/Dev_modisco_results.hdf5","r") # from Google Drive
    if task == 'Hk_contrib_scores': hdf5_results = h5py.File(outdir+"/Hk_modisco_results.hdf5","r")

    metacluster_names = [
        x.decode("utf-8") for x in 
        list(hdf5_results["metaclustering_results"]["all_metacluster_names"][:])]

    all_patterns = []

    if task == 'Dev_contrib_scores': background = np.mean(X_cand_dev, axis=(0,1))
    if task == 'Hk_contrib_scores': background = np.mean(X_cand_hk, axis=(0,1))

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
        for pattern_name in all_pattern_names[:4]:
            plt.figure(figsize=(20, 2))
            print(metacluster_name, pattern_name)
            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            print("total seqlets:",len(pattern["seqlets_and_alnmts"]["seqlets"]))
            print("IC-scaled, forwarded")
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                            background=background)) 
            plt.title(f"Metacluster: {metacluster_name}, Pattern: {pattern_name}")
            plt.savefig(outdir+f"metacluster{metacluster_name}_pattern{pattern_name}.png", dpi=300, bbox_inches='tight')
            plt.close() 

    hdf5_results.close()

def my_tfmodisco(task):
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
    return tfmodisco_results

def main(config, indir, mdldir, outdir):
    global data
    data = pd.read_table(indir)
    global main_params
    main_params = LoadConfig(config)
    X_test, Y_test = prepare_input(data[data['set'] == "Test"])
    with CustomObjectScope({'mse': MeanSquaredError}):
        global main_model
        main_model = load_model(mdldir)

    np.random.seed(seed=1234)
    background = X_test[np.random.choice(X_test.shape[0], 1000, replace=False)]
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough # this is required due to conflict between versions (https://github.com/slundberg/shap/issues/1110)
    global explainer_dev
    explainer_dev = shap.DeepExplainer((main_model.layers[0].input, main_model.layers[-2].output),
                                    data=background)
    global explainer_hk
    explainer_hk = shap.DeepExplainer((main_model.layers[0].input, main_model.layers[-1].output),
                                    data=background)
    cand_dev = data.loc[(data['set'] == 'Test') &
                        (data['ID'].str.contains('_\+_positive_peaks')) & 
                        (data['Dev_log2_enrichment'] > 3)]
    cand_dev = cand_dev.sort_values('Dev_log2_enrichment', ascending = False)
    global X_cand_dev
    X_cand_dev, Y_cand_dev = prepare_input(cand_dev)
    pred_values_dev = main_model.predict(X_cand_dev)[0].squeeze() # 0 output for developmental
    shap_values_dev = explainer_dev.shap_values(X_cand_dev)[0]
    shap_values_dev = np.expand_dims(shap_values_dev.squeeze(-1), axis=0).repeat(619, axis=0)
    final_contr_scores_dev = shap_values_dev*X_cand_dev

    cand_hk = data.loc[(data['set'] == 'Test') &
                        (data['ID'].str.contains('_\+_positive_peaks')) & 
                        (data['Hk_log2_enrichment'] > 3)]

    cand_hk = cand_hk.sort_values('Hk_log2_enrichment', ascending = False)
    global X_cand_hk
    X_cand_hk, Y_cand_hk = prepare_input(cand_hk)
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
    global task_to_scores
    task_to_scores = OrderedDict()
    global task_to_hyp_scores
    task_to_hyp_scores = OrderedDict()
    for task in tasks:
        task_to_scores[task] = [np.array(x) for x in f['contrib_scores'][task]]
        task_to_hyp_scores[task] = [np.array(x) for x in f['hyp_contrib_scores'][task]]
    f.close()

    global null_per_pos_scores
    import modisco
    reload(modisco)
    null_per_pos_scores = modisco.coordproducers.LaplaceNullDist(num_to_samp=5000)

    dev_tfmodisco_results = my_tfmodisco('Dev_contrib_scores')
    hk_tfmodisco_results = my_tfmodisco('Hk_contrib_scores')

    import modisco.util
    reload(modisco.util)
    grp = h5py.File(outdir+'Dev_modisco_results.hdf5', "w")
    dev_tfmodisco_results.save_hdf5(grp)
    grp.close()

    import modisco.util
    reload(modisco.util)
    grp = h5py.File(outdir+'Hk_modisco_results.hdf5', "w")
    hk_tfmodisco_results.save_hdf5(grp)
    grp.close()

    modisco_motif_plots('Dev_contrib_scores')
    modisco_motif_plots('Hk_contrib_scores')


if (len(sys.argv)!=5):
    exit("<parms.config> <mdl-dir> <data-indir> <outdir> \n")

(config, indir, mdldir, outdir)=sys.argv[1:]
main(config, indir, mdldir, outdir)