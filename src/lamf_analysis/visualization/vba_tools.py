
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm

from scipy.stats import spearmanr
from scipy.stats import kruskal
from scipy.stats import ttest_ind
from scipy.stats import sem
from scipy.sparse import csgraph
from scipy.stats import chisquare
from scipy import stats

from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import pairwise_distances
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

# from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache as bpc
# cache_dir = loading.get_analysis_cache_dir()
# cache = bpc.from_s3_cache(cache_dir)

# import visual_behavior_glm.GLM_analysis_tools as gat  # to get recent glm results
# import visual_behavior_glm.GLM_params as glm_params

# import umap
import random
from scipy import signal
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import seaborn as sns

from dask import delayed, compute
from dask.distributed import Client


########################################################
## VBA dimensionality reduction > clustering > plotting
def plot_clustered_dropout_scores(df, labels=None, cmap='RdBu_r', model_output_type='', ax=None,):
    sort_order = np.argsort(labels)
    sorted_labels = labels[sort_order]
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 12))
    ax = sns.heatmap(df.abs().values[sort_order], cmap='Blues', ax=ax, vmin=0, vmax=1,
                     robust=True,
                     cbar_kws={"drawedges": False, "shrink": 0.8, "label": 'fraction change in explained variance'})
    ax.set_ylabel('cells')
    ax.set_xlabel('features')
    ax.set_title('sorted dropout scores')

    ax.set_ylim(0, df.shape[0])
    ax.set_xlim(0, df.shape[1])
    ax.set_xticks(np.arange(0.5, df.shape[1] + 0.5))
    ax.set_xticklabels([key[1] + ' - ' + key[0] for key in list(df.keys())], rotation=90)
    ax.set_xticklabels(df.keys(), rotation=90)

    ax.set_xlabel('')
    ax2 = ax.twinx()
    ax2.set_yticks([0, len(df)])
    ax2.set_yticklabels([0, len(df)])

    # plot horizontal lines to seperate the clusters
    cluster_divisions = np.where(np.diff(sorted_labels) == 1)[0]
    for y in cluster_divisions:
        ax.hlines(y, xmin=0, xmax=df.shape[1], color='k')

    # set cluster labels
    cluster_divisions = np.hstack([0, cluster_divisions, len(labels)])
    mid_cluster_divisions = []
    for i in range(0, len(cluster_divisions) - 1):
        mid_cluster_divisions.append(((cluster_divisions[i + 1] - cluster_divisions[i]) / 2) + cluster_divisions[i])

    # label cluster ids
    unique_cluster_ids = np.sort(np.unique(labels))
    ax.set_yticks(mid_cluster_divisions)
    ax.set_yticklabels(unique_cluster_ids)
    ax.set_ylabel('cluster ID')

    # separate regressors
    for x in np.arange(0, df.shape[0], 3):
        ax.vlines(x, ymin=0, ymax=df.shape[0], color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig, ax


def get_elbow_plots(X, n_clusters=range(2, 20), ax=None):
    '''
    Computes within cluster density and variance explained for Kmeans method.
    :param X: data
    :param n_clusters: default = range(2, 20)
    :param ax: default=None
    :return:
    '''

    km = [KMeans(n_clusters=k).fit(X) for k in n_clusters]  # get Kmeans for different Ks
    centroids = [k.cluster_centers_ for k in km]  # get centroids of clusters

    D_k = [cdist(X, cent, 'euclidean') for cent in centroids]  # compute distance of each datapoint
    # to its centroid, n clusters by n points
    dist = [np.min(D, axis=1) for D in D_k]
    avgWithinSS = [sum(d) / X.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X)**2) / X.shape[0]
    bss = tss - wcss

    if ax is None:
        fig, ax = plt.figure(2, 1, figsize=(8, 12))

    # elbow curve

    ax[0].plot(n_clusters, avgWithinSS, 'k*-')
    ax[0].grid(True)
    ax[0].set_xlabel('Number of clusters')
    ax[0].set_ylabel('Average within-cluster sum of squares')
    ax[0].set_title('Elbow for KMeans clustering')

    ax[1].plot(n_clusters, bss / tss * 100, 'k*-')
    ax[1].grid(True)
    ax[1].set_xlabel('Number of clusters')
    ax[1].set_ylabel('Percentage of variance explained')
    ax[1].set_title('Elbow for KMeans clustering')


def plot_gap_statistic(gap_statistic, n_clusters=None, tag='', save_dir=None, folder=None):

    suffix = '_' + tag
    data = gap_statistic

    figsize = (10, 3)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    x = len(data['gap'])
    ax[0].plot(np.arange(1, x + 1), data['reference_inertia'], 'o-')
    ax[0].plot(np.arange(1, x + 1), data['ondata_inertia'], 'o-')
    ax[0].legend(['reference inertia', 'ondata intertia'], fontsize='x-small')
    ax[0].set_ylabel('Natural log of euclidean \ndistance values')
    ax[0].set_xlabel('Number of clusters')
    if n_clusters is not None:
        ax[0].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

    ax[1].plot(np.arange(1, x + 1), data['gap'], 'o-')
    ax[1].set_ylabel('Gap statistic')
    ax[1].set_xlabel('Number of clusters')
    if n_clusters is not None:
        ax[1].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')
    # ax[1].set_ylim([0, 0.3])

    plt.subplots_adjust(wspace=0.4)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'Gap_' + suffix)
    return ax


def plot_gap_statistic_with_sem(gap_statistics, n_clusters=None, tag='', save_dir=None, folder=None):

    suffix = '_' + tag
    x = len(gap_statistics['gap'])

    figsize = (10, 4)
    fig, ax = plt.subplots(1, 2, figsize=figsize)

    ax[0].fill_between(x=np.arange(1, x + 1),
                    y1=gap_statistics['reference_inertia'] + gap_statistics['reference_sem'],
                    y2=gap_statistics['reference_inertia'] - gap_statistics['reference_sem'],
                    label='reference inertia')
    ax[0].plot(np.arange(1, x + 1), gap_statistics['reference_inertia'], 'o-', label='reference inertia')
    ax[0].fill_between(x=np.arange(1, x + 1),
                    y1=gap_statistics['ondata_inertia'] + gap_statistics['ondata_sem'],
                    y2=gap_statistics['ondata_inertia'] - gap_statistics['ondata_sem'],
                    label='ondata inertia')
    ax[0].plot(np.arange(1, x + 1), gap_statistics['ondata_inertia'], 'o-', label='ondata inertia')
    ax[0].set_ylabel('Natural log of euclidean \ndistance values')
    ax[0].set_xlabel('Number of clusters')
    if n_clusters is not None:
        ax[0].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

    ax[1].fill_between(x=np.arange(1, x + 1),
                    y1=np.asarray(gap_statistics['gap']) + np.asarray(gap_statistics['gap_sem']),
                    y2=np.asarray(gap_statistics['gap']) - np.asarray(gap_statistics['gap_sem']),
                    )
    ax[1].set_ylabel('Gap statistic')
    ax[1].set_xlabel('Number of clusters')
    if n_clusters is not None:
        ax[1].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

    plt.tight_layout()

    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'Gap_' + suffix, formats=['.png', '.pdf'] )


def plot_eigengap_values(eigenvalues, n_clusters=None, save_dir=None, folder=None):

    figsize = (10,3)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    ax[0].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, '-o')
    ax[0].set_ylabel('Eigen values \n(sorted)')
    ax[0].set_xlabel('Eigen number')
    ax[0].set_xlim([0, 20])
    if n_clusters is not None:
        ax[0].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

    ax[1].plot(np.arange(2, len(eigenvalues) + 1), np.diff(eigenvalues), '-o')
    ax[1].set_ylabel('Eigengap value \n(difference)')
    ax[1].set_xlabel('Eigen number')
    ax[1].set_xlim([0, 20])
    ax[1].set_ylim([0, 0.10])
    if n_clusters is not None:
        ax[1].axvline(x=n_clusters, ymin=0, ymax=1, linestyle='--', color='gray')

    plt.subplots_adjust(wspace=0.4)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'eigengap' + suffix)
    return ax


def plot_silhouette_scores(X=None, model=KMeans, silhouette_scores=None, silhouette_std=None,
                           n_clusters=np.arange(2, 10), metric=None, n_boots=20, ax=None,
                           model_output_type=''):

    assert X is not None or silhouette_scores is not None, 'must provide either data to cluster or recomputed scores'
    assert X is None or silhouette_scores is None, 'cannot provide data to cluster and silhouette scores'

    if silhouette_scores is None:
        if metric is None:
            metric = 'euclidean'
        silhouette_scores, silhouette_std = get_silhouette_scores(X=X, model=model, n_clusters=n_clusters, metric=metric, n_boots=n_boots)
    elif silhouette_scores is not None:
        if len(silhouette_scores) != len(n_clusters):
            n_clusters = np.arange(1, len(silhouette_scores)+1)

        if metric is None:
            metric = ''

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(5,5))

    ax.plot(n_clusters, silhouette_scores, 'ko-')
    if silhouette_std is not None:
        ax.errorbar(n_clusters, silhouette_scores, silhouette_std, color='k')
    ax.set_title('{}, {}'.format(model_output_type, metric), fontsize=16)
    ax.set_xlabel('Number of clusters')
    ax.set_ylabel('Silhouette score')
    plt.grid()
    plt.tight_layout()

    return ax


def plot_umap_with_labels(X, labels, ax=None, filename_string=''):
    fit = umap.UMAP()
    u = fit.fit_transform(X)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.scatter(u[:, 0], u[:, 1], c=labels)
    plt.tight_layout()
    ax.set_title(filename_string)
    return ax


def plot_coclustering_matrix_sorted_by_cluster_size(coclustering_df, cluster_meta, 
                                                    save_dir=None, folder=None, suffix='', ax=None):
    """
    plot co-clustering matrix sorted by cluster size for a given cre_line
    will save plot if save_dir and folder are provided (and ax is None)
    if ax is provided, will plot on provided ax
    """
    cluster_meta = cluster_meta.sort_values(by='cluster_id')
    sorted_cell_specimen_ids = cluster_meta.index.values
    # sort rows and cols of coclustering matrix by sorted cell_specimen_ids
    sorted_coclustering_matrix = coclustering_df.loc[sorted_cell_specimen_ids]
    sorted_coclustering_matrix = sorted_coclustering_matrix[sorted_cell_specimen_ids]

    if ax is None:
        figsize = (8, 8)
        fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(sorted_coclustering_matrix, cmap="Greys", ax=ax, square=True,
                     cbar=True, cbar_kws={"drawedges": False, "label": 'probability of\nco-clustering', 'shrink': 0.7, },)

    ax.set_yticks((0, sorted_coclustering_matrix.shape[0]))
    ax.set_yticklabels((0, sorted_coclustering_matrix.shape[0]), fontsize=20)
    ax.set_ylabel('cells', fontsize=20)
    ax.set_xticks((0, sorted_coclustering_matrix.shape[0]))
    ax.set_xticklabels((0, sorted_coclustering_matrix.shape[0]), fontsize=20, rotation=0)
    ax.set_xlabel('')
    sns.despine(ax=ax, bottom=False, top=False, left=False, right=False)
    if save_dir:
        filename = 'coclustering_matrix_sorted_by_cluster_size_' + suffix
        utils.save_figure(fig, figsize, save_dir, folder, filename)  # saving to PDF is super slow
    return ax


def plot_clusters(dropout_df, cluster_df=None, plot_difference=False, mean_response_df=None, save_plots=False, path=None):
    '''
    Plots heatmaps and descriptors of clusters.
    dropout_df: dataframe of dropout scores, n cells by n regressors by experience level
    cluster_df: df with cluster id and cell specimen id columns
    plot_difference: Boolean to plot difference (cluster - population) of mean dropout scores on the heatmap
    mean_response_df: dataframe with mean responseswith cell specimen id, timestamps, and mean_response columns

    :return:
    '''

    # Set up the variables

    cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
    n_clusters = len(cluster_ids)
    palette = utils.get_cre_line_colors()
    palette_exp = get_experience_level_colors()
    depths = [75, 175, 275, 375]
    areas = ['VISp', 'VISl']

    # get number of animals per cluster
    # grouped_df = cluster_df.groupby('cluster_id')
    # N_mice = grouped_df.agg({"mouse_id": "nunique"})

    # Set up figure
    fig, ax = plt.subplots(5, n_clusters, figsize=(n_clusters * 2, 14),
                           sharex='row',
                           sharey='row',
                           gridspec_kw={'height_ratios': [3, 2, 2, 1, 2]},
                           )
    ax = ax.ravel()
    for i, cluster_id in enumerate(cluster_ids):

        # 1. Mean dropout scores
        this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
        mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
        if plot_difference is False:
            ax[i] = sns.heatmap(mean_dropout_df,
                            cmap='RdBu',
                            vmin=-1,
                            vmax=1,
                            ax=ax[i],
                            cbar=False, )
        elif plot_difference is True:
            mean_dropout_df_diff = mean_dropout_df.abs() - dropout_df.mean().unstack().abs()
            ax[i] = sns.heatmap(mean_dropout_df_diff,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                ax=ax[i],
                                cbar=False, )

        # 2. By cre line
        # % of total cells in this cluster
        within_cluster_df = \
            cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'cell_type').count()[['cluster_id']]
        n_cells = within_cluster_df.sum().values[0]
        all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('cell_type').count()[['cluster_id']]
        fraction_cre = within_cluster_df / all_df
        fraction_cre.sort_index(inplace=True)
        # add numerical column for sns.barplot
        fraction_cre['cell_type_index'] = np.arange(0, fraction_cre.shape[0])
        fraction = np.round(fraction_cre[['cluster_id']].values * 100, 1)

        ax[i + len(cluster_ids)] = sns.barplot(data=fraction_cre,
                                               y='cluster_id',
                                               x='cell_type_index',
                                               palette=palette,
                                               ax=ax[i + len(cluster_ids)])
        if fraction_cre.shape[0] == 3:
            ax[i + len(cluster_ids)].set_xticklabels(['Exc', 'SST', 'VIP'], rotation=90)
        # else:
        #    ax[i + len(cluster_ids)].set_xticklabels(fraction_cre.index.values, rotation=90)
        ax[i + len(cluster_ids)].set_ylabel('fraction cells\nper class')
        ax[i + len(cluster_ids)].set_xlabel('')
        # ax[i+ len(cluster_ids)].set_title('n mice = ' + str(N_mice.loc[cluster_id].values), fontsize=16)

        # set title and labels
        ax[i].set_title('cluster ' + str(i) + '\n' + str(fraction[0][0]) + '%, n=' + str(n_cells), fontsize=16)
        ax[i].set_yticklabels(mean_dropout_df.index.values, rotation=0)
        ax[i].set_ylim(-0.5, 4.5)
        ax[i].set_xlabel('')

        # 3. Plot by depth
        within_cluster_df = \
            cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'binned_depth').count()[['cluster_id']]
        all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('binned_depth').count()[['cluster_id']]
        fraction_depth = within_cluster_df / all_df
        fraction_depth.reset_index(inplace=True)
        ax[i + (len(cluster_ids) * 2)] = sns.barplot(data=fraction_depth,
                                                     x='binned_depth',
                                                     y='cluster_id',
                                                     order=depths,
                                                     palette='gray',
                                                     ax=ax[i + (len(cluster_ids) * 2)])
        # set labels
        ax[i + (len(cluster_ids) * 2)].set_xlabel('depth (um)')
        ax[i + (len(cluster_ids) * 2)].set_ylabel('fraction cells\nper depth')
        ax[i + (len(cluster_ids) * 2)].set_xticks(np.arange(0, len(depths)))
        ax[i + (len(cluster_ids) * 2)].set_xticklabels(depths, rotation=90)

        # 4. Plot by area
        within_cluster_df = \
            cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'targeted_structure').count()[['cluster_id']]
        all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('targeted_structure').count()[['cluster_id']]
        fraction_area = within_cluster_df / all_df
        fraction_area.reset_index(inplace=True, drop=False)

        ax[i + (len(cluster_ids) * 3)] = sns.barplot(data=fraction_area,
                                                     # orient='h',
                                                     x='targeted_structure',
                                                     y='cluster_id',
                                                     order=areas,
                                                     palette='gray',
                                                     ax=ax[i + (len(cluster_ids) * 3)])
        # set labels
        ax[i + (len(cluster_ids) * 3)].set_xlabel('area')
        ax[i + (len(cluster_ids) * 3)].set_ylabel('fraction cells\nper area')
        ax[i + (len(cluster_ids) * 3)].set_xticklabels(areas, rotation=0)

        # plot mean traces

        # axes_column = 'cluster_id'
        hue_column = 'experience_level'
        hue_conditions = np.sort(cluster_df[hue_column].unique())
        timestamps = cluster_df['trace_timestamps'][0]
        xlim_seconds = [-1, 1.5]
        change = False
        omitted = True
        xlabel = 'time (sec)'

        for c, hue in enumerate(hue_conditions):

            traces = cluster_df[(cluster_df['cluster_id'] == cluster_id) &
                                (cluster_df[hue_column] == hue)].mean_trace.values
            for t in range(0, np.shape(traces)[0]):
                traces[t] = signal.resample(traces[t], len(timestamps))
            ax[i + (len(cluster_ids) * 4)] = utils.plot_mean_trace(np.asarray(traces),
                                                                   timestamps, ylabel='response',
                                                                   legend_label=hue,
                                                                   color=palette_exp[c],
                                                                   interval_sec=1,
                                                                   plot_sem=False,
                                                                   xlim_seconds=xlim_seconds,
                                                                   ax=ax[i + (len(cluster_ids) * 4)])
            ax[i + (len(cluster_ids) * 4)] = utils.plot_flashes_on_trace(ax[i + (len(cluster_ids) * 4)], timestamps,
                                                                         change=change, omitted=omitted)
            ax[i + (len(cluster_ids) * 4)].axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
            ax[i + (len(cluster_ids) * 4)].set_title('')
            ax[i + (len(cluster_ids) * 4)].set_xlim(xlim_seconds)
            ax[i + (len(cluster_ids) * 4)].set_xlabel(xlabel)

        if i != 0:
            ax[i + len(cluster_ids)].set_ylabel('')
            ax[i + (len(cluster_ids) * 2)].set_ylabel('')
            ax[i + (len(cluster_ids) * 3)].set_ylabel('')
            ax[i + (len(cluster_ids) * 4)].set_ylabel('')
    return fig
    plt.tight_layout()



def plot_clusters_columns_all_cre_lines(df, df_meta, labels_cre, multi_session_df, save_dir=None):
    """
    plots dropout scores per cluster, per cre line, in a compact format
    df: reshaped dataframe of dropout scores for matched cells
    df_meta: metadata for each cell in df
    labels_cre: dict of cluster labels for each cre line
    multi_session_df: dataframe with trial averaged traces to plot per cluster
    """
    cells_table = loading.get_cell_table()
    cre_lines = np.sort(df_meta.cre_line.unique())[::-1]

    for cre_line in cre_lines:
        cell_type = df_meta[df_meta.cre_line == cre_line].cell_type.unique()[0]

        print(cre_line, cell_type)
        cids = df_meta[df_meta['cre_line'] == cre_line]['cell_specimen_id']
        df_sel = df.loc[cids]
        labels = labels_cre[cre_line]
        cluster_df = pd.DataFrame(index=df_sel.index, columns=['cluster_id'], data=labels)
        cluster_df = cluster_df.merge(cells_table[['cell_specimen_id', 'cell_type', 'cre_line', 'experience_level',
                                                   'binned_depth', 'targeted_structure']], on='cell_specimen_id')
        cluster_df = cluster_df.drop_duplicates(subset='cell_specimen_id')
        cluster_df.reset_index(inplace=True)

        dropout_df = df_sel.copy()

        cluster_mdf = multi_session_df.merge(cluster_df[['cell_specimen_id', 'cluster_id']],
                                             on='cell_specimen_id',
                                             how='inner')

        # Set up the variables

        # cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
        cluster_ids = cluster_df.groupby(['cluster_id']).count()[['cell_specimen_id']].sort_values(by='cell_specimen_id').index.values
        n_clusters = len(cluster_ids)
        # palette = utils.get_cre_line_colors()
        palette_exp = get_experience_level_colors()
        depths = [75, 175, 275, 375]
        areas = ['VISp', 'VISl']

        # get number of animals per cluster
        # grouped_df = cluster_df.groupby('cluster_id')
        # N_mice = grouped_df.agg({"mouse_id": "nunique"})

        # Set up figure
        n_cols = 4
        figsize = (12, n_clusters * 2,)
        fig, ax = plt.subplots(n_clusters, n_cols, figsize=figsize, sharex='col', gridspec_kw={'width_ratios': [2, 3, 2, 1.25]})
        ax = ax.ravel()
        for i, cluster_id in enumerate(cluster_ids[::-1]):

            # % of total cells in this cluster
            within_cluster_df = cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'cell_type').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('cell_type').count()[['cluster_id']]
            n_cells = within_cluster_df.cluster_id.values[0]
            fraction = within_cluster_df / all_df
            fraction = np.round(fraction.cluster_id.values[0] * 100, 1)

            # 1. Mean dropout scores
            this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
            mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
            ax[(i * n_cols)] = sns.heatmap(mean_dropout_df,
                                           cmap='RdBu',
                                           vmin=-1,
                                           vmax=1,
                                           ax=ax[(i * n_cols)],
                                           cbar=False, )

            # set title and labels
        #     ax[(i*n_cols)].set_title('cluster ' + str(i) + '\n' + str(fraction[0][0]) + '%, n=' + str(n_cells),  fontsize=14)
            ax[(i * n_cols)].set_title(str(fraction) + '%, n=' + str(n_cells), fontsize=14)
            ax[(i * n_cols)].set_yticks(np.arange(0.5, 4.5))
            ax[(i * n_cols)].set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=15)
            ax[(i * n_cols)].set_ylim(0, 4)
            ax[(i * n_cols)].set_xlabel('')
            ax[(i * n_cols)].set_ylabel('cluster ' + str(i), fontsize=15)

            # 3. Plot by depth
            n_col = 2
            within_cluster_df = \
                cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                    'binned_depth').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('binned_depth').count()[['cluster_id']]
            fraction_depth = within_cluster_df / all_df
            fraction_depth.reset_index(inplace=True)
            ax[(i * n_cols) + n_col] = sns.barplot(data=fraction_depth, orient='h',
                                                   y='binned_depth',
                                                   x='cluster_id',
                                                   # orient='h',
                                                   palette='gray',
                                                   ax=ax[(i * n_cols) + n_col])
            # set labels
            ax[(i * n_cols) + n_col].set_ylabel('depth (um)', fontsize=15)
            ax[(i * n_cols) + n_col].set_xlabel('')
            ax[(i * n_cols) + n_col].set_ylim(-0.5, len(depths) - 0.5)

            ax[(i * n_cols) + n_col].set_yticks(np.arange(0, len(depths)))
            ax[(i * n_cols) + n_col].set_yticklabels(depths, rotation=0)
            ax[(i * n_cols) + n_col].invert_yaxis()

            # 4. Plot by area
            n_col = 3
            within_cluster_df = \
                cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                    'targeted_structure').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('targeted_structure').count()[['cluster_id']]
            fraction_area = within_cluster_df / all_df
            fraction_area.reset_index(inplace=True, drop=False)

            ax[(i * n_cols) + n_col] = sns.barplot(data=fraction_area,
                                                   # orient='h',
                                                   x='targeted_structure',
                                                   y='cluster_id',
                                                   palette='gray',
                                                   ax=ax[(i * n_cols) + n_col])
            # set labels
            ax[(i * n_cols) + n_col].set_xlabel('')
            ax[(i * n_cols) + n_col].set_ylabel('fraction cells\nper area', fontsize=15)
            ax[(i * n_cols) + n_col].set_xticklabels(areas, rotation=0)

            # plot mean traces

            n_col = 1
            # axes_column = 'cluster_id'
            hue_column = 'experience_level'
            hue_conditions = np.sort(cluster_mdf[hue_column].unique())
            timestamps = cluster_mdf['trace_timestamps'][0]
            xlim_seconds = [-1, 1.5]
            change = False
            omitted = True
            # xlabel = 'time (sec)'

            for c, hue in enumerate(hue_conditions):

                traces = cluster_mdf[(cluster_mdf['cluster_id'] == cluster_id) &
                                     (cluster_mdf[hue_column] == hue)].mean_trace.values
                for t in range(0, np.shape(traces)[0]):
                    traces[t] = signal.resample(traces[t], len(timestamps))
                ax[(i * n_cols) + n_col] = utils.plot_mean_trace(np.asarray(traces),
                                                                 timestamps, ylabel='response',
                                                                 legend_label=hue,
                                                                 color=palette_exp[c],
                                                                 interval_sec=1,
                                                                 plot_sem=False,
                                                                 xlim_seconds=xlim_seconds,
                                                                 ax=ax[(i * n_cols) + n_col])
                ax[(i * n_cols) + n_col] = utils.plot_flashes_on_trace(ax[(i * n_cols) + n_col], timestamps,
                                                                       change=change, omitted=omitted)
                ax[(i * n_cols) + n_col].axvline(x=0, ymin=0, ymax=1, linestyle='--', color='gray')
                ax[(i * n_cols) + n_col].set_title('')
                ax[(i * n_cols) + n_col].set_xlim(xlim_seconds)
                ax[(i * n_cols) + n_col].set_xlabel('')
                ax[(i * n_cols) + n_col].set_ylabel('response', fontsize=15)

        ax[(i * n_cols) + 2].set_xlabel('fraction cells\nper depth')
        ax[(i * n_cols) + 3].set_xlabel('area')
        ax[(i * n_cols) + 1].set_xlabel('time (sec)')

        plt.suptitle(cell_type, x=0.54, y=1.01)
        plt.tight_layout()
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, 'clustering', cre_line + '_clusters_column')


def plot_clusters_compact_all_cre_lines(df, df_meta, labels_cre, save_dir=None):
    """
    plots dropout scores per cluster, per cre line, in a compact format
    df: reshaped dataframe of dropout scores for matched cells
    df_meta: metadata for each cell in df
    labels_cre: dict of cluster labels for each cre line
    """
    cells_table = loading.get_cell_table()
    cre_lines = np.sort(df_meta.cre_line.unique())[::-1]

    for cre_line in cre_lines:
        cell_type = df_meta[df_meta.cre_line == cre_line].cell_type.unique()[0]

        print(cre_line, cell_type)
        # cids = df_meta[df_meta['cre_line'] == cre_line]['cell_specimen_id']
        # df_sel = df.loc[cids]
        df_sel = df[cre_line]
        labels = labels_cre[cre_line]
        cluster_df = pd.DataFrame(index=df_sel.index, columns=['cluster_id'], data=labels)
        cluster_df = cluster_df.merge(cells_table[['cell_specimen_id', 'cell_type', 'cre_line', 'experience_level',
                                                   'binned_depth', 'targeted_structure']], on='cell_specimen_id')
        cluster_df = cluster_df.drop_duplicates(subset='cell_specimen_id')
        cluster_df.reset_index(inplace=True)

        dropout_df = df_sel.copy()

        # Set up the variables

        # cluster_ids = cluster_df['cluster_id'].value_counts().index.values  # sort cluster ids by size
        cluster_ids = cluster_df.groupby(['cluster_id']).count()[['cell_specimen_id']].sort_values(by='cell_specimen_id').index.values
        n_clusters = len(cluster_ids)

        # get number of animals per cluster
        # grouped_df = cluster_df.groupby('cluster_id')
        # N_mice = grouped_df.agg({"mouse_id": "nunique"})

        # Set up figure
        n_rows = int(np.ceil(n_clusters / 3.))
        figsize = (5, 1.9 * n_rows)
        fig, ax = plt.subplots(n_rows, 3, figsize=figsize, sharex=True, sharey=True)
        ax = ax.ravel()
        for i, cluster_id in enumerate(cluster_ids[::-1]):

            # % of total cells in this cluster
            within_cluster_df = cluster_df[cluster_df['cluster_id'] == cluster_id].drop_duplicates('cell_specimen_id').groupby(
                'cell_type').count()[['cluster_id']]
            all_df = cluster_df.drop_duplicates('cell_specimen_id').groupby('cell_type').count()[['cluster_id']]
            n_cells = within_cluster_df.cluster_id.values[0]
            fraction = within_cluster_df / all_df
            fraction = np.round(fraction.cluster_id.values[0] * 100, 1)

            # 1. Mean dropout scores
            this_cluster_ids = cluster_df[cluster_df['cluster_id'] == cluster_id]['cell_specimen_id'].unique()
            mean_dropout_df = dropout_df.loc[this_cluster_ids].mean().unstack()
            ax[i] = sns.heatmap(mean_dropout_df,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                ax=ax[i],
                                cbar=False, )

            # set title and labels
            ax[i].set_title('cluster ' + str(i) + '\n' + str(fraction) + '%, n=' + str(n_cells), fontsize=14)
            ax[i].set_yticks(np.arange(0.5, 4.5))
            ax[i].set_yticklabels(mean_dropout_df.index.values, rotation=0, fontsize=15)
            ax[i].set_ylim(0, 4)
            ax[i].set_xlabel('')

        plt.suptitle(cell_type, x=0.5, y=1.01)
        plt.subplots_adjust(wspace=0.4, hspace=0.8)
        if save_dir:
            utils.save_figure(fig, figsize, save_dir, 'clustering', cre_line + '_clusters_compact')


def get_cluster_colors(labels):
    '''
    generates a list of unique colors for each cluster for plots
    :param labels: list of cluster labels, length of N cells
    :return: label_colors: a list of strings for color names
    '''

    color_map = get_cluster_color_map(labels)
    label_colors = []
    for label in labels:
        label_colors.append(color_map[label])
    return label_colors


def get_cluster_color_map(labels):
    '''
    # generates a dictionary of cluster label: color
    :param labels: list of cluster labels, length of N cells
    :return: dictionary color number: color string
    '''
    unique_labels = np.sort(np.unique(labels))
    color_map = {}
    # r = int(random.random() * 256)
    # g = int(random.random() * 256)
    # b = int(random.random() * 256)
    # step = 256 / len(unique_labels)
    for i in unique_labels:
        # r += step
        # g += step
        # b += step
        # r = int(r) % 256
        # g = int(g) % 256
        # b = int(b) % 256
        # color_map[i] = (r, g, b)
        color_map[i] = '#%06X' % random.randint(0, 0xFFFFFF)
    return color_map


def plot_N_clusters_by_cre_line(labels_cre, ax=None, palette=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    if palette is None:
        palette = [(1.0, 0.596078431372549, 0.5882352941176471),
                   (0.6196078431372549, 0.8549019607843137, 0.8980392156862745),
                   (0.7725490196078432, 0.6901960784313725, 0.8352941176470589)]
    for i, cre_line in enumerate(labels_cre.keys()):
        labels = labels_cre[cre_line]
        (unique, counts) = np.unique(labels, return_counts=True)
        sorted_counts = np.sort(counts)[::-1]
        frequencies = sorted_counts / sum(counts) * 100
        cumulative_sum = [0]
        for freq in frequencies:
            cumulative_sum.append(cumulative_sum[-1] + freq)
        ax.grid()
        ax.plot(range(0, len(cumulative_sum)), cumulative_sum, 'o-', color=palette[i],
                linewidth=4, markersize=10)
        ax.set_xlabel('Cluster number')
        ax.set_ylabel('Cells per cluster (%)')
    ax.legend(['Excitatory', 'SST inhibitory', 'VIP inhibitory'])

    return ax


def plot_cluster_density(df_dropouts=None, labels_list=None, cluster_corrs=None, ax=None):
    if cluster_corrs is None:
        cluster_corrs = get_cluster_density(df_dropouts, labels_list)
    df_labels = pd.DataFrame(columns=['corr', 'labels'])
    for key in cluster_corrs.keys():
        data = np.array([cluster_corrs[key], np.repeat(key, len(cluster_corrs[key]))])
        tmp = pd.DataFrame(data.transpose(), columns=['corr', 'labels'])
        df_labels = df_labels.append(tmp, ignore_index=True)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(max(cluster_corrs.keys()), 6))
    ax.axhline(xmin=-1, xmax=max(cluster_corrs.keys()) + 1, color='grey')
    # sns.violinplot(data=df_labels, x='labels', y='corr', ax=ax)
    sns.boxplot(data=df_labels, x='labels', y='corr', ax=ax)
    return ax


########################################################
## VBA dimensionality reduction > clustering > processing
def load_eigengap(glm_version, feature_matrix, cell_metadata=None, save_dir=None, k_max=25):
    """
    if eigengap values were computed and file exists in save_dir, load it
    otherwise run get_eigenDecomposition for a range of 1 to k_max clusters
    returns dictionary of eigengap for each cre line = [nb_clusters, eigenvalues, eigenvectors]
    # this doesnt actually take too long, so might not be a huge need to save files besides records
    """
    eigengap_filename = f'eigengap_{glm_version:2}_k_max_{k_max}.pkl'
    eigengap_path = os.path.join(save_dir, eigengap_filename)
    if os.path.exists(eigengap_path):
        print('loading eigengap values scores from', eigengap_path)
        with open(eigengap_path, 'rb') as f:
            eigengap = pickle.load(f)
            f.close()
        print('done')
    else:
        X = feature_matrix.values
        sc = SpectralClustering(2)  # N of clusters does not impact affinity matrix
        # but you can obtain affinity matrix only after fitting, thus some N of clusters must be provided.
        sc.fit(X)
        A = sc.affinity_matrix_
        eigenvalues, eigenvectors, nb_clusters = get_eigenDecomposition(A, max_n_clusters=k_max)
        eigengap = [nb_clusters, eigenvalues, eigenvectors]
        save_clustering_results(eigengap, filename_string=eigengap_filename, path=save_dir)
    return eigengap


def get_eigenDecomposition(A, max_n_clusters=25):
    """
    Input:
    A: Affinity matrix from spectral clustering
    max_n_clusters

    :return A tuple containing:
    - the optimal number of clusters by eigengap heuristic
    - all eigen values
    - all eigen vectors

    This method performs the eigen decomposition on a given affinity matrix,
    following the steps recommended in the paper:
    1. Construct the normalized laplacian matrix: L = D−1/2ADˆ −1/2.
    2. Find the eigenvalues and their associated eigen vectors
    3. Identify the maximum gap which corresponds to the number of clusters
    by eigengap heuristic

    References:
    https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
    """
    L = csgraph.laplacian(A, normed=True)
    # n_components = A.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # Identify the optimal number of clusters as the index corresponding
    # to the larger gap between eigen values
    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:max_n_clusters]
    nb_clusters = index_largest_gap + 1

    return eigenvalues, eigenvectors, nb_clusters


def get_silhouette_scores(X, model=SpectralClustering, n_clusters=np.arange(2, 10), metric='euclidean', n_boots=20):
    '''
    Computes silhouette scores for given n clusters.
    :param X: data, n observations by n features
    :param model: default = SpectralClustering, but you can pass any clustering model object (some models might have unique
                    parameters, so this might cause issues in the future)
    :param n_clusters: an array or list of number of clusters to use.
    :param metric: default = 'euclidean', distance metric to use inner and outer cluster distance
                    (other options in scipy.spatial.distance.pdist)
    :param n_boots: default = 20, number of repeats to average over for each n cluster
    _____________
    :return: silhouette_scores: a list of scores for each n cluster
    '''
    print('size of X = ' + str(np.shape(X)))
    print('NaNs in the array = ' + str(np.sum(X == np.nan)))
    silhouette_scores = []
    silhouette_std = []
    for n_cluster in n_clusters:
        s_tmp = []
        for n_boot in range(0, n_boots):
            model.n_clusters = n_cluster
            md = model.fit(X)
            try:
                labels = md.labels_
            except AttributeError:
                labels = md
            s_tmp.append(silhouette_score(X, labels, metric=metric))
        silhouette_scores.append(np.mean(s_tmp))
        silhouette_std.append(np.std(s_tmp))
        print('n {} clusters mean score = {}'.format(n_cluster, np.mean(s_tmp)))
    return silhouette_scores, silhouette_std


def get_labels_for_coclust_matrix(X, model=SpectralClustering, nboot=np.arange(100), n_clusters=8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use. Object must be initialized.
    :param nboot: (list or an array) default = 100, number of clustering repeats
    :param n_clusters: (num) default = 8
    ___________
    :return: labels: matrix of labels, n repeats by n observations
    '''
    if model is SpectralClustering:
        model = model()
    labels = []
    if n_clusters is not None:
        model.n_clusters = n_clusters
    for _ in tqdm(nboot):
        md = model.fit(X)
        labels.append(md.labels_)
    return labels


def get_coClust_matrix(X, model=SpectralClustering, nboot=np.arange(150), n_clusters=8):
    '''

    :param X: (ndarray) data, n observations by n features
    :param model: (clustering object) default =  SpectralClustering; clustering method to use. Model must be initialized.
    :param nboot: (list or an array) default = 100, number of clustering repeats
    :param n_clusters: (num) default = 8
    ______________
    returns: coClust_matrix: (ndarray) probability matrix of co-clustering together.
    '''
    # model = model()
    labels = get_labels_for_coclust_matrix(X=X,
                                           model=model,
                                           nboot=nboot,
                                           n_clusters=n_clusters)
    coClust_matrix = []
    for i in range(np.shape(labels)[1]):  # get cluster id of this observation
        this_coClust_matrix = []
        for j in nboot:  # find other observations with this cluster id
            id = labels[j][i]
            this_coClust_matrix.append(labels[j] == id)
        coClust_matrix.append(np.sum(this_coClust_matrix, axis=0) / max(nboot))
    return coClust_matrix


def clean_cells_table(cells_table=None, columns=None, add_binned_depth=True):
    '''
    Adds metadata to cells table using ophys_experiment_id. Removes NaNs and duplicates
    :param cells_table: vba loading.get_cell_table()
    :param columns: columns to add, default = ['cre_line', 'imaging_depth', 'targeted_structure']
    :return: cells_table with selected columns
    '''

    if cells_table is None:
        cells_table = loading.get_cell_table()

    if columns is None:
        columns = ['cre_line', 'imaging_depth', 'targeted_structure', 'binned_depth', 'cell_type']

    cells_table = cells_table[columns]

    # drop NaNs
    cells_table.dropna(inplace=True)

    # drop duplicated cells
    cells_table.drop_duplicates('cell_specimen_id', inplace=True)

    # set cell specimen ids as index
    cells_table.set_index('cell_specimen_id', inplace=True)

    if add_binned_depth is True and 'imaging_depth' in cells_table.keys():
        depths = [75, 175, 275, 375]
        cells_table['binned_depth'] = np.nan
        for i, imaging_depth in enumerate(cells_table['imaging_depth']):
            for depth in depths[::-1]:
                if imaging_depth < depth:
                    cells_table['binned_depth'].iloc[i] = depth

    # print number of cells
    print('N cells {}'.format(len(cells_table)))

    return cells_table


def save_clustering_results(data, filename_string='', path=None):
    '''
    for HCP scripts to save output of spectral clustering in a specific folder
    :param data: what to save
    :param filename_string: name of the file, use as descriptive info as possible
    :return:
    '''
    if path is None:
        path = '/root/capsule/scratch/cluster_results/'

    if os.path.exists(path) is False:
        os.mkdir(path)

    filename = os.path.join(path, '{}'.format(filename_string))
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    f.close()


def get_cre_line_cell_specimen_ids(df):
    cre_lines = df.cre_line.unique()
    cre_line_ids = {}
    for cre_line in cre_lines:
        ids = df[df.cre_line == cre_line]['cell_specimen_id'].unique()
        cre_line_ids[cre_line] = ids
    return cre_line_ids


def kruskal_by_experience_level(df_pivoted, posthoc=True):
    stats = {}
    f = df_pivoted['Familiar'].values
    n = df_pivoted['Novel 1'].values
    nn = df_pivoted['Novel >1'].values

    k, p = kruskal(f, n, nn)
    stats['KW'] = (k, p)
    if posthoc:
        t, p = ttest_ind(f, n, nan_policy='omit')
        stats['Familiar_vs_Novel'] = (t, p)
        t, p = ttest_ind(f, nn, nan_policy='omit')
        stats['Familiar_vs_Novel>1'] = (t, p)
        t, p = ttest_ind(n, nn, nan_policy='omit')
        stats['Novel_vs_Novel>1'] = (t, p)

    return stats


def pivot_df(df, dropna=True, drop_duplicated_cells=True):
    df_pivoted = df.groupby(['cell_specimen_id', 'experience_level']).mean()
    if dropna is True:
        df_pivoted = df_pivoted.unstack().dropna()
    else:
        df_pivoted = df_pivoted.unstack()

    if drop_duplicated_cells is True:
        if len(df) == len(np.unique(df.index.values)):
            print('No duplicated cells found')
        elif len(df) > len(np.unique(df.index.values)):
            print('found {} duplicated cells. But not removed. This needs to be fixed'.format(len(df) - len(np.unique(df.index.values))))
        elif len(df) < len(np.unique(df.index.values)):
            print('something weird happened!!')

    return df_pivoted


def build_stats_table(metrics_df, metrics_columns=None, dropna=True, pivot=False):
    # check for cre lines
    if 'cre_line' in metrics_df.keys():
        cre_lines = metrics_df['cre_line'].unique()
        cre_line_ids = get_cre_line_cell_specimen_ids(metrics_df)
    else:
        cre_lines = ['all']
        cre_line_ids = metrics_df['cell_specimen_id'].unique()

    # get selected columns
    if metrics_columns is None:
        metrics_columns = ['image_selectivity_index', 'image_selectivity_index_one_vs_all',
                           'lifetime_sparseness', 'fraction_significant_p_value_gray_screen',
                           'fano_factor', 'reliability', 'running_modulation_index']
        if 'hit_miss_index' in metrics_df.keys():
            metrics_columns = [*metrics_columns, 'hit_miss_index']

    # check which columns are in the dataframe
    metrics_columns_corrected = []
    for metric in metrics_columns:
        if metric in metrics_df.keys():
            metrics_columns_corrected.append(metric)

    stats_table = pd.DataFrame(columns=['cre_line', 'comparison', 'statistic', 'metric', 'data'])
    statistics = ('t', 'pvalue')
    for c, cre_line in enumerate(cre_lines):
        # dummy table
        if cre_line == 'all':
            tmp_cre = metrics_df
        else:
            tmp_cre = metrics_df[metrics_df['cell_specimen_id'].isin(cre_line_ids[cre_line])]

        # group df by cell id and experience level
        metrics_df_pivoted = pivot_df(tmp_cre, dropna=dropna)
        for m, metric in enumerate(metrics_columns_corrected):
            stats = kruskal_by_experience_level(metrics_df_pivoted[metric])
            for i, stat in enumerate(statistics):
                for key in stats.keys():
                    data = {'cre_line': cre_line,
                            'comparison': key,
                            'statistic': stat,
                            'metric': metric,
                            'data': stats[key][i], }
                    tmp_table = pd.DataFrame(data, index=[0])
                    stats_table = stats_table.append(tmp_table, ignore_index=True)

    if pivot and stats_table['data'].sum() != 0:
        stats_table = pd.pivot_table(stats_table, columns=['metric'],
                                     index=['cre_line', 'comparison', 'statistic']).unstack()
    elif stats_table['data'].sum() == 0:
        print('Cannot pivot table when all data is NaNs')
    return stats_table


def get_cluster_density(df_dropouts, labels_list, use_spearmanr=False):
    '''
    Computes correlation coefficients for clusters computed on glm dropouts
    '''
    labels = np.unique(labels_list)
    cluster_corrs = {}
    for label in labels:
        cluster_means = df_dropouts[labels_list == label].mean().abs().values
        within_cluster = df_dropouts[labels_list == label].abs()
        corr_coeffs = []
        for cid in within_cluster.index.values:
            if use_spearmanr is False:
                corr_coeffs.append(np.corrcoef(cluster_means, within_cluster.loc[cid].values)[0][1])
            elif use_spearmanr is True:
                corr_coeffs.append(spearmanr(cluster_means, within_cluster.loc[cid].values)[0])
        cluster_corrs[label] = corr_coeffs
    return cluster_corrs


def shuffle_dropout_score(df_dropout, shuffle_type='all'):
    '''
    Shuffles dataframe with dropout scores from GLM.
    shuffle_type: str, default='all', other options= 'experience', 'regressors', 'experience_within_cell',
                        'full_experience'

    Returns:
        df_shuffled (pd. Dataframe) of shuffled dropout scores
    '''
    
    df_shuffled = df_dropout.copy()
    regressors = df_dropout.columns.levels[0].values
    experience_levels = df_dropout.columns.levels[1].values
    if shuffle_type == 'all':
        # print('shuffling all data')
        for column in df_dropout.columns:
            df_shuffled[column] = df_dropout[column].sample(frac=1).values

    elif shuffle_type == 'experience':
        print('shuffling data across experience')
        assert np.shape(df_dropout.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        for experience_level in experience_levels:
            randomized_cids = df_dropout.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for regressor in regressors:
                    df_shuffled.iloc[i][(regressor, experience_level)] = df_dropout.loc[cid][(regressor, experience_level)]

    elif shuffle_type == 'regressors':
        print('shuffling data across regressors')
        assert np.shape(df_dropout.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        for regressor in regressors:
            randomized_cids = df_dropout.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for experience_level in experience_levels:
                    df_shuffled.iloc[i][(regressor, experience_level)] = df_dropout.loc[cid][(regressor, experience_level)]

    elif shuffle_type == 'experience_within_cell':
        print('shuffling data across experience within each cell')
        cids = df_dropout.index.values
        experience_level_shuffled = experience_levels.copy()
        for cid in cids:
            np.random.shuffle(experience_level_shuffled)
            for j, experience_level in enumerate(experience_level_shuffled):
                for regressor in regressors:
                    df_shuffled.loc[cid][(regressor, experience_levels[j])] = df_dropout.loc[cid][(regressor,
                                                                                                    experience_level)]
    elif shuffle_type == 'full_experience':
        print('shuffling data across experience fully (cell id and experience level)')
        assert np.shape(df_dropout.columns.levels)[
            0] == 2, 'df should have two level column structure, 1 - regressors, 2 - experience'
        # Shuffle cell ids first
        for experience_level in experience_levels:
            randomized_cids = df_dropout.sample(frac=1).index.values
            for i, cid in enumerate(randomized_cids):
                for regressor in regressors:
                    df_shuffled.iloc[i][(regressor, experience_level)] = df_dropout.loc[cid][
                        (regressor, experience_level)]
        # Shuffle experience labels
        df_shuffled_again = df_shuffled.copy(deep=True)
        cids = df_shuffled.index.values
        experience_level_shuffled = experience_levels.copy()
        for cid in cids:
            np.random.shuffle(experience_level_shuffled)
            for j, experience_level in enumerate(experience_level_shuffled):
                for regressor in regressors:
                    df_shuffled_again.loc[cid][(regressor, experience_levels[j])] = df_shuffled.loc[cid][(regressor,
                                                                                                            experience_level)]

        df_shuffled = df_shuffled_again.copy()

    else:
        print('no such shuffle type..')
        df_shuffled = None
    return df_shuffled


def compute_inertia(a, X, metric = 'euclidean'):
    W = [np.mean(pairwise_distances(X[a == c, :], metric=metric)) for c in np.unique(a)]
    return np.mean(W)


def compute_gap(clustering, data, k_max=5, n_boots=20, reference_shuffle='all', metric='euclidean',
                parallel=False):
    '''
    Computes gap statistic between clustered data (ondata inertia) and null hypothesis (reference intertia).

    :param clustering: clustering object that includes "n_clusters" and "fit_predict"
    :param data: an array of data to be clustered (n samples by n features)
    :param k_max: (int) maximum number of clusters to test, starts at 1
    :param n_boots: (int) number of repetitions for computing mean inertias
    :param reference: (str) what type of shuffle to use, shuffle_dropout_scores,
            None is use random normal distribution
    :param metric: (str) type of distance to use, default = 'euclidean'
    :param parallel: (bool) default = False, use parallel computing to speed up the process
    
    :return:
    gap: array of gap values that are the difference between two inertias
    reference_inertia: array of log of reference inertia
    ondata_inertia: array of log of ondata inertia
    
    TODO: can be improved using parallel computing
    dask parallelization at the level of within n_boots made it slower.
    CPU usage near full without parallelization
    '''

    if len(data.shape) == 1:
        data = data.reshape(-1, 1)

    if isinstance(data, pd.core.frame.DataFrame):
        data_array = data.values
    else:
        data_array = data

    gap_statistics = {}
    reference_inertia = []
    reference_sem = []
    gap_mean = []
    gap_sem = []
    
    def _compute_inertia_shuffled(k, data, reference_shuffle, clustering, metric):
        # draw random dist or shuffle
        if reference_shuffle is None:
            reference = np.random.rand(*data.shape) * -1
        else:
            reference_df = shuffle_dropout_score(data, shuffle_type=reference_shuffle)
            reference = reference_df.values

        clustering.n_clusters = k
        assignments = clustering.fit_predict(reference)
        return compute_inertia(assignments, reference, metric=metric)
    
    for k in range(1, k_max):
        print(f'Reference inertia for {k} clusters')
        if parallel:
            task = []
            with Client() as client:
                for _ in range(n_boots):
                     task.append(delayed(_compute_inertia_shuffled)(k, data, reference_shuffle, clustering, metric))
                local_ref_inertia = compute(task)
        else:
            local_ref_inertia = []
            for _ in range(n_boots):
                local_ref_inertia.append(_compute_inertia_shuffled(k, data, reference_shuffle, clustering, metric))
        reference_inertia.append(np.mean(local_ref_inertia))
        reference_sem.append(sem(local_ref_inertia))

    def _compute_ondata_inertia(k, data, clustering, metric):
        clustering.n_clusters = k
        assignments = clustering.fit_predict(data)
        return compute_inertia(assignments, data, metric=metric)
    
    ondata_inertia = []
    ondata_sem = []
    for k in range(1, k_max):
        print(f'On data inertia for {k} clusters')
        if parallel:
            task = []
            with Client() as client:
                for _ in range(n_boots):
                    task.append(delayed(_compute_ondata_inertia)(k, data_array, clustering, metric))
                local_ondata_inertia = compute(task)
        else:
            local_ondata_inertia = []
            for _ in range(n_boots):
                local_ondata_inertia.append(_compute_ondata_inertia(k, data_array, clustering, metric))
        ondata_inertia.append(np.mean(local_ondata_inertia))
        ondata_sem.append(sem(local_ondata_inertia))

        # compute difference before mean
        gap_mean.append(np.mean(np.subtract(np.log(local_ondata_inertia), np.log(local_ref_inertia))))
        gap_sem.append(sem(np.subtract(np.log(local_ondata_inertia), np.log(local_ref_inertia))))

    # maybe plotting error bars with this metric would be helpful but for now I'll leave it
    gap = np.log(reference_inertia) - np.log(ondata_inertia)

    # we potentially do not need all of this info but saving it to plot it for now
    gap_statistics['gap'] = gap
    gap_statistics['reference_inertia'] = np.log(reference_inertia)
    gap_statistics['ondata_inertia'] = np.log(ondata_inertia)
    gap_statistics['reference_sem'] = reference_sem
    gap_statistics['ondata_sem'] = ondata_sem
    gap_statistics['gap_mean'] = gap_mean
    gap_statistics['gap_sem'] = gap_sem

    return gap_statistics


##################################
## VBA visualization utils


def get_experience_level_colors():
    """
    get color map corresponding to Familiar, Novel 1 and Novel >1
    Familiar = blue
    Novel = red
    Novel+1 = purple
    """
    
    blues = sns.color_palette('Blues_r', 6)[:5][::2]
    reds = sns.color_palette('Reds_r', 6)[:5][::2]
    purples = sns.color_palette('Purples_r', 6)[:5][::2]

    colors = [blues[0], reds[0], purples[0]]

    return colors


def get_conditions_string(data_type, conditions):
    """
    creates a string containing the data_type and conditions corresponding to a given multi_session_df.
    ignores first element in conditions which is usually 'cell_specimen_id' or 'ophys_experiment_id'
    :param data_type: 'events', 'filtered_events', 'dff'
    :param conditions: list of conditions used to group for averaging in multi_session_df
                        ex: ['cell_specimen_id', 'is_change', 'image_name'], or ['cell_specimen_id', 'engagement_state', 'omitted']
    """

    if len(conditions) == 6:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + \
                            conditions[4] + '_' + conditions[5]
    elif len(conditions) == 5:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3] + '_' + \
                            conditions[4]
    elif len(conditions) == 4:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2] + '_' + conditions[3]
    elif len(conditions) == 3:
        conditions_string = data_type + '_' + conditions[1] + '_' + conditions[2]
    elif len(conditions) == 2:
        conditions_string = data_type + '_' + conditions[1]
    elif len(conditions) == 1:
        conditions_string = data_type + '_' + conditions[0]

    return conditions_string


def remap_coding_scores_to_session_colors(coding_scores):
    """
    coding_scores is an array where rows are cells (or cluster ids) and columns are experience level / coding feature combinations
    """
    import matplotlib
    coding_scores_remapped = coding_scores.copy()


    colors = get_experience_level_colors()
    # colors = c_vals # Kyle's colors
    coding_score_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", colors[0], "white", colors[1],
                                                                                 "white", colors[2]])

    # familiar sessions are in scale of 0-1 already
    # add 2 to novel sessions to make them in the scale of colors[1]
    coding_scores_remapped.loc[:, (slice(None), "Novel")] += 2
    coding_scores_remapped.loc[:, (slice(None), "Novel +")] += 4

    # return max value for plotting
    vmax = 5

    return coding_scores_remapped, coding_score_cmap, vmax


def plot_feature_matrix_sorted(feature_matrix, cluster_meta, sort_col='cluster_id', use_abbreviated_labels=False,
                               resort_by_size=False, cmap='Blues', vmax=1, save_dir=None, folder=None, suffix=''):
    """
    plots feature matrix used for clustering sorted by sort_col

    sort_col: column in cluster_meta to sort rows of feature_matrix (cells) by
    """
    # check if there are negative values in feature_matrix, if so, use diff cmap and set vmin to -1
    if len(np.where(feature_matrix < 0)[0]) > 0:
        vmin = -1
    else:
        vmin = 0

    figsize = (15, 5)
    fig, ax = plt.subplots(1, 3, figsize=figsize)
    for i, cre_line in enumerate(get_cre_lines(cluster_meta)):
        cluster_meta_cre = cluster_meta[cluster_meta.cre_line == cre_line]
        # get cell ids for this cre line in sorted order
        if resort_by_size:
            cluster_size_order = cluster_meta_cre['cluster_id'].value_counts().index.values
            cluster_meta_cre['size_sort_cluster_id'] = [np.where(cluster_size_order == label)[0][0] for label in
                                                  cluster_meta_cre.cluster_id.values]
            sort_col = 'size_sort_cluster_id'
        sorted_cluster_meta_cre = cluster_meta_cre.sort_values(by=sort_col)
        cell_order = sorted_cluster_meta_cre.index.values
        label_values = sorted_cluster_meta_cre[sort_col].values

        # get data from feature matrix for this set of cells
        data = feature_matrix.loc[cell_order]
        ax[i] = sns.heatmap(data.values, cmap=cmap, ax=ax[i], vmin=vmin, vmax=vmax,
                            robust=True, cbar_kws={"drawedges": False, "shrink": 0.7, "label": 'coding score'})

        for x in [3, 6, 9]:
            ax[i].axvline(x=x, ymin=0, ymax=data.shape[0], color='gray', linestyle='--', linewidth=1)
        ax[i].set_title(get_cell_type_for_cre_line(cre_line, cluster_meta))
        ax[i].set_ylabel('cells')
        ax[i].set_ylim(0, data.shape[0])
        ax[i].set_yticks([0, data.shape[0]])
        ax[i].set_yticklabels((0, data.shape[0]), fontsize=14)
        ax[i].set_ylim(ax[i].get_ylim()[::-1])  # flip y axes so larger clusters are on top
        ax[i].set_xlabel('')
        ax[i].set_xlim(0, data.shape[1])
        ax[i].set_xticks(np.arange(0, data.shape[1]) + 0.5)
        if use_abbreviated_labels:
            xticklabels = [get_abbreviated_experience_levels([key[1]])[0] + ' -  ' + get_abbreviated_features([key[0]])[0].upper() for key in list(data.keys())]
            ax[i].set_xticklabels(xticklabels, rotation=90, fontsize=14)
        else:
            ax[i].set_xticklabels([key[1] + ' -  ' + key[0] for key in list(data.keys())], rotation=90, fontsize=14)

        # plot a line at the division point between clusters
        cluster_divisions = np.where(np.diff(label_values) == 1)[0]
        for y in cluster_divisions:
            ax[i].hlines(y, xmin=0, xmax=data.shape[1], color='k')

    fig.subplots_adjust(wspace=0.7)
    if save_dir:
        utils.save_figure(fig, figsize, save_dir, folder, 'feature_matrix_sorted_by_' + sort_col + suffix)
        
        
def plot_flashes_on_trace(ax, timestamps, change_time=0, change=None, omitted=False, alpha=0.075, facecolor='gray'):
    """
    plot stimulus flash durations on the given axis according to the provided timestamps
    """
    stim_duration = 0.25
    blank_duration = 0.5
    start_time = timestamps[0]
    end_time = timestamps[-1]
    interval = (blank_duration + stim_duration)
    # after time 0
    if omitted:
        array = np.arange((change_time + interval), end_time, interval) # image array starts at the next interval
        # plot a dashed line where the stimulus time would have been
        ax.axvline(x=change_time, ymin=0, ymax=1, linestyle='--', color=sns.color_palette()[9], linewidth=1.5)
    else:
        array = np.arange(change_time, end_time, interval)
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        if change and (i == 0):
            change_color = sns.color_palette()[0]
            ax.axvspan(amin, amax, facecolor=change_color, edgecolor='none', alpha=alpha*1.5, linewidth=0, zorder=1)
        else:
            ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    # if change == True:
    #     alpha = alpha / 2.
    # else:
    #     alpha
    # before time 0
    array = np.arange(change_time, start_time, -interval)
    array = array[1:]
    for i, vals in enumerate(array):
        amin = array[i]
        amax = array[i] + stim_duration
        ax.axvspan(amin, amax, facecolor=facecolor, edgecolor='none', alpha=alpha, linewidth=0, zorder=1)
    return ax


def plot_mean_trace(traces, timestamps, ylabel='dF/F', legend_label=None, color='k', interval_sec=1, xlim_seconds=[-2, 2],
                    plot_sem=True, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    if len(traces) > 0:
        trace = np.mean(traces, axis=0)
        sem = (np.std(traces)) / np.sqrt(float(len(traces)))
        ax.plot(timestamps, trace, label=legend_label, linewidth=2, color=color)
        if plot_sem:
            ax.fill_between(timestamps, trace + sem, trace - sem, alpha=0.5, color=color)
        ax.set_xticks(np.arange(int(timestamps[0]), int(timestamps[-1]) + 1, interval_sec))
        ax.set_xlim(xlim_seconds)
        ax.set_xlabel('time (sec)')
        ax.set_ylabel(ylabel)
    sns.despine(ax=ax)
    return ax


def save_figure(fig, figsize, save_dir, folder, fig_title, formats=['.png']):
    fig_dir = os.path.join(save_dir, folder)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    filename = os.path.join(fig_dir, fig_title)
    mpl.rcParams['pdf.fonttype'] = 42
    fig.set_size_inches(figsize)
    for f in formats:
        fig.savefig(filename + f, transparent=True, orientation='landscape', bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())
