from .session_stim_response import SessionStimResponse
from .stim_response import stim_response_all
from .bod_utils import load_bods_for_session

from aind_ophys_data_access import file_handling

from pathlib import Path

def save_session_stim_response(session_bods,
                               output_dir = Path("../scratch")):
    """Save SessionStimResponse h5 files for all planes in a session.
    
    Parameters
    ----------
    session_bods : dict
        Dictionary of BehaviorOphysDataset objects for a session.
        key: plane_name, value: BehaviorOphysDataset object
    output_dir : Path
        Directory to save output files.
        
    Returns
    -------
    None
    """
    stim_response_dict = {}
    
    # check all same session
    session_names = [bod.metadata["processed_path"].name for bod in session_bods.values()]
    assert len(set(session_names)) == 1, "All BehaviorOphysDatasets must come from the same session"
    
    for plane, bod in session_bods.items():
        
        # calculate stim response data frames for all combos of event/data type
        sr_dict = stim_response_all(bod, full_time_window=False)
        
        # make dict keys in the form (plane, event_type, data_type)
        for key, value in sr_dict.items():
            new_key = (plane,) + key
            stim_response_dict[new_key] = value
        
    # session name shenanigans
    session_name = bod.metadata["processed_path"].name
    mouse_id, session_date = session_name.split("_")[1:3]
    processed_sec = session_name.split("_")[-1]
    session_name_short = f"{mouse_id}_{session_date}"
    
    # remove empty values from dict
    stim_response_dict = {k: v for k, v in stim_response_dict.items() if v is not None}

    # save to .h5 using fancy class
    session_stim_response = SessionStimResponse(data_dict=stim_response_dict,
                                                session_name=session_name)
    output_path = output_dir / f"{session_name_short}_session_stim_response.h5"
    print(f"Saving to {output_path}")
    session_stim_response.save(output_path)
    
    
def save_all_session_stim_response_for_mouse(mouse_id: str, 
                                             output_dir: Path("../scratch/"),
                                             bod_kwargs: dict = {}):

    files= file_handling.all_sessions_in_capsule()
    assert mouse_id in files.keys(), f"Mouse {mouse_id} not found in capsule, attach data"
    mouse_files = files[mouse_id]
    
    for session_name in mouse_files.keys():
        session_bods = load_bods_for_session(mouse_files, 
                                             session_name, 
                                             bod_kwargs=bod_kwargs)
        if session_bods is not None:
            try:
                save_session_stim_response(session_bods, output_dir)
            except Exception as e:
                logging.error(f"Failed to save stim response for session {session_name} with error: {e}")
        del session_bods
