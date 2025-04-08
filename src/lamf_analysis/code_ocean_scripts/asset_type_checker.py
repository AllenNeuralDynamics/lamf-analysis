"""

# AssetTypeChecker Class
+ The goal of this class is to identify raw data assets that are missing a particular kind
of derived assets. For example, multiplane-ophys sessions might have "processed", "dlc-eye",
"dlc-side", "stim_response", etc. In some cases, we collect cortical stacks on certain sessions;
We need to identify those sessions and make sure that the cortical stacks were generated.
+ In this case, we should check the files on S3. Then compared to existing derived assets.
+ Return a list of sessions that have the raw data but no derived assets of a particular type.
+ We use the aind_session library to get the sessions.

## Example usage
from asset_checker import AssetTypeChecker

### Initialize checker for cortical z-stack files
checker = AssetTypeChecker(
    file_pattern="*cortical_z_stack*",
    subfolder="pophys",  # Restrict search to ophys subfolder
    processed_asset_types=["processed"]  # Check for processed assets
)

### Check all multiplane-ophys sessions
results = checker.check_sessions(platform="multiplane-ophys")

### Get matching sessions
matching_sessions = checker.get_matching_sessions()
print(f"Found {len(matching_sessions)} sessions with cortical z-stack files and processed assets")

### View results as DataFrame for easy analysis
df = checker.to_dataframe()
print(df.head())

### Save results to CSV
df.to_csv("cortical_z_stack_results.csv", index=False)
"""

from pathlib import Path
import json
import pandas as pd

# import lamf_analysis.code_ocean_scripts.jobs as jobs
# import aind_ophys_data_access.session_utils as session_utils
# import codeocean

import aind_session


from typing import List, Dict, Any
import s3fs
from tqdm import tqdm

class AssetTypeChecker:
    """
    Class to check for specific file types in session asset buckets.
    
    This class helps identify sessions that contain specific files in their
    S3 storage locations and/or have specific processed asset types.
    """
    
    def __init__(self, 
                 file_pattern: str = None, 
                 subfolder: str = None, 
                 processed_asset_types: List[str] = None,
                 verbose: bool = True):
        """
        Initialize the AssetTypeChecker.

        Credentials
        -----------
        The credentials are stored in the ~/.aws/credentials file.
        
        Parameters:
        -----------
        file_pattern : str, optional
            File pattern to search for in S3 buckets (e.g., "*cortical_z_stack*")
        subfolder : str, optional
            Subfolder within bucket to restrict search (e.g., "pophys")
        processed_asset_types : List[str], optional
            List of asset types to check for in session.data_assets
        verbose : bool, default=True
            Whether to print progress information
        """
        self.file_pattern = file_pattern
        self.subfolder = subfolder
        self.processed_asset_types = processed_asset_types or []
        self.verbose = verbose
        
        # Initialize S3 filesystem
        self.s3 = s3fs.S3FileSystem(anon=False)
        
        # Results storage
        self.results = {}
        self.bucket_cache = {}  # Cache bucket listings for performance
    
    
    def check_session(self, session: aind_session.Session) -> Dict[str, Any]:
        """
        Check if a session contains the specified file pattern and/or asset types.
        
        Parameters:
        -----------
        session : aind_session.Session
            Session object to check
            
        Returns:
        --------
        Dict[str, Any]
            Result dictionary with match information
        """
        result = {
            "session_id": session.id,
            "file_matches": [],
            "asset_type_matches": []
        }
        
        # Get S3 location from session metadata
        try:
            #s3_location = session.docdb["location"] # both give the same location
            #print(f"S3 location: {s3_location}")
            s3_location = str(session.raw_data_dir)
            #print(f"S3 location: {s3_location}")
            result["s3_location"] = s3_location
        except (KeyError, AttributeError, FileNotFoundError):
            if self.verbose:
                print(f"S3 location not found for session {session.id}")
            result["file_match"] = False
            result["asset_type_match"] = False
            return result
            
        # Check for file pattern in S3 bucket
        if self.file_pattern:
            bucket_prefix = s3_location.replace("s3://", "")
            
            # If subfolder is specified, add it to the path
            if self.subfolder:
                search_path = f"{bucket_prefix}/{self.subfolder}"
            else:
                search_path = bucket_prefix
                
            # Get bucket and prefix
            parts = bucket_prefix.split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
            
            try:
                # Search for matching files
                if self.verbose:
                    print(f"Searching for {self.file_pattern} in {search_path}")
                
                # Use glob to find matching files
                matching_files = self.s3.glob(f"{search_path}/**/{self.file_pattern}")
                
                if matching_files:
                    result["file_matches"] = matching_files
                    result["file_match"] = True
                    if self.verbose:
                        print(f"Found {len(matching_files)} files matching {self.file_pattern}")    
                else:
                    result["file_match"] = False
            except Exception as e:
                if self.verbose:
                    print(f"Error searching S3 for session {session.name}: {str(e)}")
                result["file_match"] = False
                result["error"] = str(e)
        else:
            # Skip file matching if no pattern provided
            result["file_match"] = None
        
        # Check for specific asset types
        if self.processed_asset_types:
            asset_matches = []
            
            for asset in session.data_assets:
                #if asset.type.value == "result" and asset.name in self.processed_asset_types:
                if any(asset_type in asset.name for asset_type in self.processed_asset_types):
                    asset_matches.append(asset.name)
            
            result["asset_type_matches"] = asset_matches
            result["asset_type_match"] = len(asset_matches) > 0
        else:
            # Skip asset type matching if no types provided
            result["asset_type_match"] = None
            
        return result
    
    def check_sessions(self, sessions: List[aind_session.Session] = None, platform: str = "multiplane-ophys") -> Dict[str, Dict]:
        """
        Check multiple sessions for matching files and asset types.
        
        Parameters:
        -----------
        sessions : List[aind_session.Session], optional
            List of session objects to check. If None, will fetch sessions for the platform.
        platform : str, default="multiplane-ophys"
            Platform to get sessions for if sessions is None
            
        Returns:
        --------
        Dict[str, Dict]
            Dictionary mapping session names to result dictionaries
        """
        if sessions is None:
            subject_ids = ['767022', '741824', '635380', '761461', '762077', '724567', '746540', '484631', '762821', '755263', '687000', '738331', '726465', '719363', '662680', '729088', '777655', '762818', '774936', '775682', '766757', '749014', '472271', '777654', '759730', '726433', '477052', '736963', '735170', '762819', '746542', '753906', '740659', '739564', '499478', '570949', '754803', '741866', '741865', '740066', '759732', '749011', '738332', '731328', '717824', '729417', '633542', '755252', '726087', '767018', '759731', '732777', '749315', '747667', '622756', '719374', '747443', '740067', '753562', '721291', '770962', '612771', '651007', '765945', '749335', '622537', '777656', '624855', '628165', '762551', '759075', '693996', '779190', '764964', '731327', '687001', '692478', '754804', '749013', '755212', '737674', '747107', '778174', '746541', '745909', '641993', '757436', '741863', '759003']
            all_session = []
            for subject_id in subject_ids:
                print(f"Fetching sessions for subject {subject_id}")
                sessions = aind_session.get_sessions(platform=platform, subject_id=subject_id)
                all_session.extend(sessions)
            sessions = all_session
        
        results = {}
        
        # Process sessions with progress bar
        for session in tqdm(sessions, desc="Checking sessions"):
            results[session.id] = self.check_session(session)
        
        self.results = results
        return results
    
    def get_matching_sessions(self) -> List[str]:
        """
        Get names of sessions that match all criteria.
        
        Returns:
        --------
        List[str]
            List of session names that match all criteria
        """
        matching_sessions = []
        
        for session_id, result in self.results.items():
            # Check file match if requested
            file_match = True
            if self.file_pattern is not None:
                file_match = result.get("file_match", False)
            
            # Check asset type match if requested
            asset_match = True
            if self.processed_asset_types:
                asset_match = result.get("asset_type_match", False)
            
            # Add to list if all requested criteria match
            if file_match and asset_match:
                matching_sessions.append(session_id)
        
        return matching_sessions
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert results to a pandas DataFrame for easy analysis.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing check results
        """
        # Convert results to list of dictionaries for DataFrame
        records = []
        
        for session_id, result in self.results.items():
            record = {
                "session_id": session_id,
                "s3_location": result.get("s3_location", None),
                "file_match": result.get("file_match", None),
                "num_file_matches": len(result.get("file_matches", [])),
                "asset_type_match": result.get("asset_type_match", None),
                "num_asset_type_matches": len(result.get("asset_type_matches", []))
            }
            records.append(record)

        # set session_id as index
        df = pd.DataFrame(records).set_index("session_id")
        return df

    def get_summary(self) -> Dict[str, int]:
        """
        Get summary statistics of check results.
        
        Returns:
        --------
        Dict[str, int]
            Dictionary with summary statistics
        """
        df = self.to_dataframe()
        
        summary = {
            "total_sessions": len(df),
            "sessions_with_file_matches": df["file_match"].sum() if "file_match" in df.columns else 0,
            "sessions_with_asset_type_matches": df["asset_type_match"].sum() if "asset_type_match" in df.columns else 0,
            "sessions_with_both_matches": ((df["file_match"] == True) & (df["asset_type_match"] == True)).sum() 
                if ("file_match" in df.columns and "asset_type_match" in df.columns) else 0
        }
        
        return summary