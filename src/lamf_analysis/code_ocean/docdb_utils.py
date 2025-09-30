import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient


def get_docdb_api_client():
    API_GATEWAY_HOST = "api.allenneuraldynamics.org"
    DATABASE = "metadata_index"
    COLLECTION = "data_assets"
    docdb_api_client = MetadataDbClient(
            host=API_GATEWAY_HOST,
            database=DATABASE,
            collection=COLLECTION,
        )
    return docdb_api_client
    

def get_session_infos_from_docdb(subject_id, docdb_api_client=None,
                                 data_type='multiplane-ophys',
                                 filter_test_data=True):
    if docdb_api_client is None:
        docdb_api_client = get_docdb_api_client()

    query = {"subject.subject_id": subject_id, "data_description.data_level": "raw"}
    subject_response = docdb_api_client.retrieve_docdb_records(
                    filter_query=query,                
                    )
    session_infos = pd.DataFrame()
    for response in subject_response:
        # schema_version = response['schema_version']
        # if schema_version == '1.1.1': # '1.1.1' and '1.0.2' tested
        if response['name'].startswith(data_type):
            acquisition_date = response['session']['session_start_time'][:10]
            session_name = subject_id + "_" + acquisition_date
            session_type = response['session']['session_type']
            reward_consumed = response['session']['reward_consumed_total']
            rig_id = response['session']['rig_id']
            data_asset_name = response['name']
            temp_info = {"acquisition_date": acquisition_date,
                        "session_type": session_type,
                        "reward_consumed": reward_consumed,
                        "rig_id": rig_id,
                        "session_name": session_name,
                        "raw_asset_name": data_asset_name,
                        }
            session_infos = pd.concat([session_infos, pd.DataFrame(temp_info, index=[0])], ignore_index=True)
            # else:
            #     print(f"Schema version {schema_version} not handled.")
    
    session_infos.sort_values(by='acquisition_date', inplace=True)
    session_infos.reset_index(drop=True, inplace=True)
    session_infos['session_type_exposures'] = session_infos.groupby('session_type').cumcount() + 1

    if filter_test_data:
        session_infos = _filter_test_data(session_infos)

    return session_infos


def _filter_test_data(session_infos):
    ''' Any sessions after the last 3 STAGE_1 sessions are considered test data and removed.
    '''
    last_session = session_infos.query('session_type == "STAGE_1" and session_type_exposures == 3')
    assert len(last_session) == 1    
    last_acq_date = last_session['acquisition_date'].max()
    session_infos = session_infos.query('acquisition_date <= @last_acq_date').copy()
    return session_infos


def get_dff_long_baseline_window(processed_asset_ids, docdb_api_client=None):
    if docdb_api_client is None:
        docdb_api_client = get_docdb_api_client()
    assert isinstance(processed_asset_ids, list), "processed_asset_ids should be a list"
    agg_query = [
        # Match documents with "dF/F estimation" in data processes
        {
            "$match": {
                "external_links.Code Ocean": {"$in": processed_asset_ids}
            }
        },
        # Project to filter only the dF/F estimation processes
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "code_ocean_id": {"$arrayElemAt": ["$external_links.Code Ocean", 0]},  # Extract just the first Code Ocean ID as a string
                "df_f_params": {
                    "$filter": {
                        "input": "$processing.processing_pipeline.data_processes",
                        "as": "process",
                        "cond": {"$eq": ["$$process.name", "dF/F estimation"]}
                    }
                }
            }
        },
        # Only keep documents where long_window exists and is not empty
        {
            "$match": {
                "df_f_params.parameters.long_window": {"$exists": True, "$ne": []}
            }
        },
        # Project just the fields we want in the output, extracting just the first element of the array
        {
            "$project": {
                "_id": 1,
                "name": 1,
                "code_ocean_id": 1,  # Keep the Code Ocean IDs in the final projection
                "long_window": {"$arrayElemAt": ["$df_f_params.parameters.long_window", 0]}
            }
        },
    ]

    # Execute the aggregation using the client
    results = docdb_api_client.aggregate_docdb_records(agg_query)
    return results