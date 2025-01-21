
from pathlib import Path
from comb.behavior_ophys_dataset import BehaviorOphysDataset

import logging
logger = logging.getLogger(__name__)

def load_bods_for_session(session_files: dict, 
                          session_name: str,
                          bod_kwargs: dict) -> dict:
    
    """Load BehaviorOphysDataset objects for a given session from a dictionary of session files.
    
    Parameters
    ----------
    session_files : dict
        Dictionary of session files, with session name as keys and file paths as values.
    session_name : str
        Name of session to load (key in dict).
    bod_kwargs : dict
        Keyword arguments to pass to BehaviorOphysDataset constructor.

    Returns
    -------
    session_bods : dict
    """
    logging.info(f"Gathering BehaviorOpysDataset objects for session {session_name}")
    session_bods = {}
    try:
        session_file = session_files[session_name]
        raw_path = Path(session_file['raw_path'])
    
        for plane_name, plane_files in session_file['planes'].items():
            try: 
                logging.info(f"Loading plane {plane_name}")
                bod = BehaviorOphysDataset(plane_folder_path = plane_files["processed_plane_path"],
                                raw_folder_path = raw_path,
                                **bod_kwargs)
                
                # HACK
                if "cell_specimen_id" not in bod.cell_specimen_table.columns:
                    bod.cell_specimen_table["cell_specimen_id"] = bod.cell_specimen_table["cell_roi_id"]      
                session_bods[plane_name] = bod
            except Exception as e:
                logging.error(f"Failed to load plane {plane_name} with error: {e}")
                session_bods[plane_name] = None
        session_bods = {k: session_bods[k] for k in sorted(session_bods.keys())}

    except KeyError:
        logging.error(f"Could not find session {session_name} in session_files")
        return None
    
    return session_bods



def load_bods_for_session_old(session_files: dict, 
                          session_name: str,
                          **body_kwargs) -> dict:
    
    """Load BehaviorOphysDataset objects for a given session from a dictionary of session files.
    
    Parameters
    ----------
    session_files : dict
        Dictionary of session files, with session name as keys and file paths as values.
    session_name : str
        Name of session to load (key in dict).
    roi_matching_path : str
        Path to roi matching file.
    
    Returns
    -------
    session_bods : dict
    """
    logging.info(f"Gathering BehaviorOpysDataset objects for session {session_name}")
    session_bods = {}
    try:
        session_file = session_files[session_name]
        raw_path = Path(session_file['raw_path'])
    
        for plane_name, plane_files in session_file['planes'].items():
            try: 
                logging.info(f"Loading plane {plane_name}")
                bod = BehaviorOphysDataset(plane_folder_path = plane_files["processed_plane_path"],
                                raw_folder_path = raw_path,
                                project_code = project_code,
                                roi_matching_path = roi_matching_path)         
                session_bods[plane_name] = bod
            except Exception as e:
                logging.error(f"Failed to load plane {plane_name} with error: {e}")
                session_bods[plane_name] = None

    except KeyError:
        logging.error(f"Could not find session {session_name} in session_files")
        return None
    
    return session_bods