import os
import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient
from codeocean import CodeOcean


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


def get_co_client():
    domain = "https://codeocean.allenneuraldynamics.org/"
    token = os.getenv("API_SECRET")
    client = CodeOcean(domain=domain, token=token)
    return client
    

def get_session_infos_from_docdb(subject_id, docdb_api_client=None,
                                 data_type='multiplane-ophys',
                                 filter_test_data=True):
    if docdb_api_client is None:
        docdb_api_client = get_docdb_api_client()
    subject_id = str(subject_id)
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


def get_processed_data_info(mouse_id, docdb_api_client=None):
    if docdb_api_client is None:
        docdb_api_client = get_docdb_api_client()
    agg_pipeline = [
        # Match documents with 'processed' in the name (case-insensitive)
        # And ensure processing.processing_pipeline.data_processes exists
        {
            '$match': {
                'name': {'$regex': 'processed', '$options': 'i'},
                'processing.processing_pipeline.data_processes': {
                    '$exists': True,
                    '$elemMatch': {
                        "name": "dF/F estimation",
                    }
                },
                'subject.subject_id': str(mouse_id),
            }
        },
        # Project to include name and count of data_processes
        {
            '$project': {
                'name': 1,
                '_id': 1,
                'external_links': 1,
                'long_window': {
                    '$let': {
                        'vars': {
                            'df_processes': {
                                '$filter': {
                                    'input': '$processing.processing_pipeline.data_processes',
                                    'as': 'process',
                                    'cond': {'$eq': ['$$process.name', 'dF/F estimation']}
                                }
                            }
                        },
                        'in': {'$arrayElemAt': ['$$df_processes.parameters.long_window', 0]}
                    }
                }
            }
        },
        {
            '$limit': 1000
        }
    ]

    results = docdb_api_client.aggregate_docdb_records(pipeline=agg_pipeline)

    results_df = pd.DataFrame(results)
    results_df['data_asset_id'] = results_df['external_links'].apply(lambda x: x['Code Ocean'][0])
    results_df['processed_date'] = results_df['name'].str.split('_').str[-2]
    results_df['raw_name'] = results_df['name'].str.split('_processed_').str[0]

    results_df = results_df[['raw_name', 'long_window', 'data_asset_id', 'processed_date', 'name' ]]

    return results_df


def filter_data_asset_info_by_date(data_asset_info, cutoff_date='2025-09-02'):
    ''' Filter data_asset_info DataFrame to only include entries with processed_date
        less than or equal to cutoff_date.
        
        Args:
            data_asset_info (pd.DataFrame): DataFrame containing data asset information.
            cutoff_date (str): Cutoff date in 'YYYY-MM-DD' format.
                Default: '2025-09-02', when correct decrosstalk was applied. (It was wrong from 2025-02-03, but before that, there was no information about long_window)
        
        Returns:
            pd.DataFrame: Filtered DataFrame.
    '''
    filtered_info = data_asset_info[data_asset_info['processed_date'] >= cutoff_date].copy()
    return filtered_info


def filter_data_asset_info_by_long_window(data_asset_info, target_long_window):
    ''' Filter data_asset_info DataFrame to only include entries with long_window
        equal to target_long_window.
        
        Args:
            data_asset_info (pd.DataFrame): DataFrame containing data asset information.
            target_long_window (int): Target long_window value.
                Either 60 or 1800 (1800 can be better for inhibitory neurons)
        
        Returns:
            pd.DataFrame: Filtered DataFrame.
    '''
    accepted_target_values = [60, 1800]
    assert target_long_window in accepted_target_values, f"target_long_window should be one of {accepted_target_values}"
    filtered_info = data_asset_info[data_asset_info['long_window'] == target_long_window].copy()
    return filtered_info


def check_exist_in_code_ocean(results_df, co_client=None):
    if co_client is None:
        co_client = get_co_client()
    for _, row in results_df.iterrows():
        data_asset_id = row['data_asset_id']
        data_asset_name = row['name']
        data_asset = co_client.data_assets.get_data_asset(data_asset_id)
        assert data_asset.name == data_asset_name, f"Data asset {data_asset_id} not in Code Ocean"
    return True