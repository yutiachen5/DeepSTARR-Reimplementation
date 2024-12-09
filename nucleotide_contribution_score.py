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
from PIL import Image

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

def nucleotide_scores(data, main_model, main_params, explainer_dev, explainer_hk, outdir):
    cand_dev_spec = data.loc[(data['set'] == 'Test') &
                        (data['ID'].str.contains('_\+_positive_peaks')) &
                        (data['Dev_log2_enrichment'] > 3) &
                        (data['Hk_log2_enrichment'] < 2)]
    cand_dev_spec = cand_dev_spec.sort_values('Dev_log2_enrichment', ascending = False)

    cand_hk_spec = data.loc[(data['set'] == 'Test') &
                        (data['ID'].str.contains('_\+_positive_peaks')) &
                        (data['Hk_log2_enrichment'] > 3) &
                        (data['Dev_log2_enrichment'] < 1)]
    cand_hk_spec = cand_hk_spec.sort_values('Dev_log2_enrichment', ascending = False)

    # combine top 10 dev- and hk-specific enhancers
    cand_specific_enh = pd.concat([cand_dev_spec.head(10),
                                cand_hk_spec.head(10)])

    # one-hot encode sequences
    X_cand_specific_enh, Y_cand_specific_enh = prepare_input(cand_specific_enh, main_params)

    # Predict activity with model [but this is from the genome-wide model...]
    pred_values_both = main_model.predict(X_cand_specific_enh)

    # calculate scores, get first element of output, combine both dev and hk
    shap_values_both = [explainer_dev.shap_values(X_cand_specific_enh)[0],
                    explainer_hk.shap_values(X_cand_specific_enh)[0]]
    shap_values_both = [np.expand_dims(d.squeeze(-1), axis=0).repeat(20, axis=0) for d in shap_values_both]

    # multiply by one-hot sequence to keep only the scores of the actual nucleotides
    final_contr_scores_both = [d*X_cand_specific_enh for d in shap_values_both]

    titles = []
    for i in [0,13]:
        enh_type = 'Dev' if cand_specific_enh.iloc[i]['Dev_log2_enrichment'] > cand_specific_enh.iloc[i]['Hk_log2_enrichment'] else 'Hk'
        # print(enh_type, 'enhancer:', cand_specific_enh.iloc[i]['ID'])
        for t in [0,1]: # dev and hk
            plt.figure(figsize=(20, 2))
            enh_class = 'Dev' if t==0 else 'Hk'
            # print(enh_class, 'scores')
            # print('Enhancer:', cand_specific_enh.iloc[i]['ID'],
            #     ' / Obs', enh_class, 'act:', '{0:0.2f}'.format(cand_specific_enh.iloc[i][str(enh_class + '_log2_enrichment')]),
            #     ' / Pred', enh_class, 'act:', '{0:0.2f}'.format(pred_values_both[t].squeeze()[i]))
            viz_sequence.plot_weights(final_contr_scores_both[t][i], figsize=(20,2), subticks_frequency=20)
            title = (
            f"{enh_type} enhancer: {cand_specific_enh.iloc[i]['ID']}"
            f"{enh_class} scores | Obs {enh_class} act: "
            f"{cand_specific_enh.iloc[i][str(enh_class + '_log2_enrichment')]:0.2f}, "
            f"Pred {enh_class} act: {pred_values_both[t].squeeze()[i]:0.2f}\n")
            titles.append(title.strip())
            plt.savefig(outdir+f"t{t}_i{i}.png", dpi=300, bbox_inches='tight')
            plt.close() 
  
    fig, axes = plt.subplots(4, 1, figsize=(20, 12))  

    idx = 0
    for t in [0,1]:
        for i in [0,13]:
            img_path = outdir+f"t{t}_i{i}.png"  
            img = Image.open(img_path)
            
            axes[idx].imshow(img)
            axes[idx].axis('off')  
            axes[idx].set_title(titles[idx], fontsize=20)  

            os.remove(img_path)
            idx += 1
    plt.tight_layout()
    plt.savefig(outdir+"nucleotide_contribution_scores_baseline.png", dpi=300, bbox_inches='tight') 

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
    nucleotide_scores(data, main_model, main_params, explainer_dev, explainer_hk, outdir)

if (len(sys.argv)!=5):
    exit("<parms.config> <mdl-dir> <data-indir> <outdir> \n")

(config, indir, mdldir, outdir)=sys.argv[1:]
main(config, indir, mdldir, outdir)