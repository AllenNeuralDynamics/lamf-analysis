# %%
from asset_type_checker import AssetTypeChecker

%reload_ext autoreload
%autoreload 2
# %%
session = aind_session.Session("multiplane-ophys_775682_2025-03-07_09-09-28")
s3_location =session.docdb["location"]

# %%
checker = AssetTypeChecker(
    file_pattern="*cortical_z_stack*",
    subfolder="pophys",  # Restrict search to ophys subfolder
    processed_asset_types=["cortical-zstack-reg", "cortical_stack_reg","cortical-stack-reg", "cortical_zstack_reg"]  # Check for processed assets
)
# %%
results = checker.check_sessions(platform="multiplane-ophys")

# %%
df = checker.to_dataframe()

# %%
df.head()

# %%
df.to_csv("cortical_stack_results.csv", index=True)

# 
# DF WHERE MATCHES FILE_MATCH TRUE ASSET_TYPE_MATCH FALSE
# %%

stack_no_asset = df[df["file_match"] & ~df["asset_type_match"]]
# %%
for session_id, row in stack_no_asset.iterrows():
    session = aind_session.Session(session_id)
    genotype = session.docdb.get("subject").get('genotype')
    mount = session.raw_data_asset.mount
    raw_asset_id = session.raw_data_asset.id
    subject_id = session.subject.id 
    stack_no_asset.at[session_id, "genotype"] = genotype
    stack_no_asset.at[session_id, "mount"] = mount
    stack_no_asset.at[session_id, "raw_asset_id"] = raw_asset_id
    stack_no_asset.at[session_id, "subject_id"] = subject_id

# %%
stack_no_asset.genotype.value_counts()
# %%
# get genotype = Slc32a1-IRES-Cre/wt;Oi1(TIT2L-jGCaMP8s-WPRE-ICL-IRES-tTA2)/wt 
table_select = stack_no_asset[stack_no_asset["genotype"] == "Slc32a1-IRES-Cre/wt;Oi1(TIT2L-jGCaMP8s-WPRE-ICL-IRES-tTA2)/wt"]

# add  
assets_dict = []
for session_id, row in table_select.iterrows():
    assets_dict.append({"id": row["raw_asset_id"], "mount": row["mount"], "subject_id": row["subject_id"]})
assets_dict
# add genotype from session.docdb
# %%
import jobs
jobs.default_cortical_zstack_registration(json_output_path="2025-03-09-cortical-zstack-reg-jobs.json",
                                          batch_assets_list=assets_dict)