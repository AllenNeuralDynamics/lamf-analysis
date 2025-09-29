import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient


def get_session_infos_from_docdb(subject_id, docdb_api_client=None, data_type='multiplane-ophys'):
    if docdb_api_client is None:
        API_GATEWAY_HOST = "api.allenneuraldynamics.org"
        DATABASE = "metadata_index"
        COLLECTION = "data_assets"
        docdb_api_client = MetadataDbClient(
                host=API_GATEWAY_HOST,
                database=DATABASE,
                collection=COLLECTION,
            )

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

    return session_infos