from pathlib import Path
import pandas as pd
import numpy as np

# requires tkinter which depends on libX11.so.6
def load_roicat_tracking_data(path):
    from roicat import util
    r = util.RichFile_ROICaT(path=path)

    try:
        results = r.load()
    except TypeError as e:
        if '_ParzenEstimatorParameters.__new__() takes 7 positional arguments but 8 were given' not in str(e):
            raise

        failed_paths = []

        type_lookup = r.type_lookup
        optuna_entry = dict(type_lookup['optuna_study'])

        def _safe_load_optuna_study(path, **kwargs):
            failed_paths.append(str(path))
            return {'_skipped': 'optuna_study_unpickle_incompatible', 'path': str(path)}

        optuna_entry['function_load'] = _safe_load_optuna_study
        type_lookup['optuna_study'] = optuna_entry
        results = r.load(type_lookup=type_lookup)

        base_path = Path(path)
        failed_data_names = []
        for p in failed_paths:
            rel = Path(p).relative_to(base_path)
            clean_parts = [part.replace('.dict_item', '').replace('.dict', '') for part in rel.parts]
            failed_data_names.append('/'.join(clean_parts))

        print('Fallback load used due to optuna pickle incompatibility.')
        print('failed_data_names:', sorted(set(failed_data_names)))
        print('failed_paths:')
        for p in failed_paths:
            print(' -', p)

    return results


def get_alignment_template_to_all(roicat_plane_path):
    aligner_path = Path(roicat_plane_path) / 'tracking.run_data.richfile/aligner.dict_item'
    r = util.RichFile_ROICaT(path=aligner_path)
    values = r.load().value
    return values['results_geometric']['final']['alignment_template_to_all']


def load_roicat_results(roicat_path):
    # get all names of subfolders in roicat_path
    roicat_path = Path(roicat_path)
    plane_folders = [f for f in roicat_path.iterdir() if f.is_dir()]
    if len(plane_folders) == 0:
        raise ValueError(f"No plane folders found in {roicat_path}")
    results_dfs = []
    for plane_folder in plane_folders:
        results_df = load_roicat_plane_results(plane_folder)
        results_dfs.append(results_df)
    combined_results_df = pd.concat(results_dfs, ignore_index=True)
    return combined_results_df


def load_roicat_plane_results(roicat_plane_path):
    results_fn = Path(roicat_plane_path) / 'ROICaT.tracking.results.csv'
    assert results_fn.is_file(), f"Results file not found: {results_fn}"
    results_df = pd.read_csv(results_fn)
    return results_df


def get_session_name_order(roicat_plane_path):
    results_df = load_roicat_plane_results(roicat_plane_path)
    return results_df.session_name.unique()


def alignment_failed_session_names(roicat_plane_path):
    roicat_plane_path = Path(roicat_plane_path)
    session_names = get_session_name_order(roicat_plane_path)
    alignment_successful = np.array(get_alignment_template_to_all(roicat_plane_path))
    failed_session_names = []
    failed_inds = np.where(alignment_successful == False)[0]
    if len(failed_inds) > 0:
        for ind in failed_inds:
            failed_session_names.append(session_names[ind])
    return failed_session_names