"""
feature_engineering.py will contain the neccessary FeatureEngineer class to perform feature engineering
"""
from datetime import datetime
import logging
import os
import pandas as pd
from tqdm import tqdm_pandas, tqdm

import hdb_resale_estimator as hdb_est

logger = logging.getLogger("__name__")
tqdm_pandas(tqdm())

class FeatureEngineer:
    """
    FeatureEngineer class contains methods to calculate/extract
    new derived features from existing features of each hdb flat
    transaction.

    Utilizes configuration parameters to generate 
    these new features
    """

    def __init__(self, params: dict, inference_mode: bool=False, directory=None)-> None:
        self.month_feature = params["month"]
        self.feature_engineering_params = params["feature_engineering"]
        self.year_feature = self.feature_engineering_params["year"]
        self.year_month_feature = self.feature_engineering_params["year_month"]
        self.inference_mode = inference_mode
        self.directory = directory

    def engineer_features(self, hdb_data: pd.DataFrame) -> pd.DataFrame:

        """
        Engineer features from cleaned hdb data

        Args:
            hdb_data (pd.DataFrame): Dataframe containing raw hdb features

        Returns:
           pd.DataFrame: Output dataframe containing
           each hdb transaction and their respective derived features
        """

        logger.info("Mapping towns to regions...")
        hdb_data = self.map_regions(hdb_data, self.feature_engineering_params["map_regions"])

        logger.info("Extracting transaction's year and month...")
        hdb_data = self.extract_year_month(hdb_data)

        logger.info("Calculating lease age...")
        hdb_data = self.calculate_lease_age(hdb_data, self.feature_engineering_params["calculate_lease_age"])

        logger.info("Generating amenity features...")
        amenity_features_list = self.generate_amenities_features(hdb_data, self.feature_engineering_params["generate_amenities_features"])

        logger.info("Merging hdb derived features...")
        derived_features_hdb = pd.concat([hdb_data]+
                                          amenity_features_list, axis = 1)
        
        if not self.inference_mode:
            derived_features_hdb["date_context"] = datetime.strptime(os.getenv("DATE"), "%Y-%m-%d")

        return derived_features_hdb
    

    def map_regions(self, hdb_data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Function to map each town to its respective region

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            params (dict): Config params

        Returns:
            pd.DataFrame: Dataframe containing each hdb transaction with regions feature
        """
        region_feature = params['region']
        town_feature  = params['town']
        map_regions = params['mapping']

        hdb_data[region_feature] = hdb_data[town_feature].map({town: region for region, towns in map_regions.items() for town in towns})

        return hdb_data
    
    def extract_year_month(self, hdb_data: pd.DataFrame) -> pd.DataFrame:
        """Function to extract the transaction's respective year and month

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction

        Returns:
            pd.DataFrame: Dataframe containing each hdb transaction with year and month features
        """

        hdb_data[self.year_month_feature] = hdb_data[self.month_feature]
        hdb_data[self.month_feature] = hdb_data[self.year_month_feature].dt.month
        hdb_data[self.year_feature] = hdb_data[self.year_month_feature].dt.year

        return hdb_data
    
    def calculate_lease_age(self, hdb_data: pd.DataFrame, params: dict) -> pd.DataFrame:
        """Function to calculate the lease age of the hdb flat

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            params (dict): Config params

        Returns:
            pd.DataFrame: Dataframe containing each hdb transaction with lease age feature
        """
        lease_age_feature = params['lease_age']
        lease_commence_date_feature = params['lease_commence_date']

        hdb_data[lease_age_feature] = hdb_data[self.year_feature] - hdb_data[lease_commence_date_feature]

        return hdb_data

    def generate_amenities_features(self, hdb_data: pd.DataFrame, params: dict) -> list:
        """Function to generate amenity features

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            params (dict): Config params

        Returns:
            list: list of dataframes containing all amenity features
        """

        latitude_feature = params['latitude']
        longitude_feature = params['longitude']
        block_feature = params['block']
        street_name_feature = params['street_name']
        amenities = params['amenities']

        logger.info("Generating lat long coordinates...")
        hdb_coordinates = pd.DataFrame(hdb_data.progress_apply(lambda x: hdb_est.utils.find_coordinates(x[block_feature] + " " + x[street_name_feature]), axis = 1).tolist(),
                                                                                                                                                            columns=[latitude_feature,
                                                                                                                                                                    longitude_feature])
        amenity_features_list = [hdb_coordinates]
        for amenity in amenities:
           logger.info(f"Getting nearest {amenity}...")
           feature_df = self.get_nearests_amenities(pd.concat([hdb_coordinates, hdb_data[self.year_month_feature]], axis = 1),
                                                              amenity,
                                                              latitude_feature,
                                                              longitude_feature,
                                                              **amenities[amenity])
           amenity_features_list.append(feature_df)

        return amenity_features_list

    def get_nearests_amenities(self,
                               hdb_coordinates: pd.DataFrame,
                               amenity: str,
                               latitude_feature: str,
                               longitude_feature: str,
                               amenities_data: dict,
                               radius: int,
                               period: bool) -> pd.DataFrame:
        """Function to get the following features for each flat:
                - no_of_amenities_within_radius
                - distance_to_nearest_amenity

        Args:
            hdb_data (pd.DataFrame): Dataframe containing each hdb transaction
            amenity (str): type of amenity (eg parks, schools, malls)
            coordinates_feature(str): name of coordinates feature
            amenities_data (dict): params to read dataframe containing the coordinates of each amenity location
            radius (int): radius around the flat
            period (bool): whether to take into account the opening date of the amenity

        Returns:
            pd.DataFrame: Dataframe containing the amenity-specific features 
        """
        source = amenities_data["read_from_source"]
        read_params = amenities_data["params"]
        if self.directory:
            data_path = read_params["data_path"]
            read_params["data_path"] = f"{self.directory}/{data_path}"
        amenity_details = hdb_est.utils.read_data(source = source, params=read_params)
        if period:
            amenity_details = amenity_details.rename(columns={"Opening year": "YEAR", "Opening month": "MONTH"})
            amenity_details[self.year_month_feature] = pd.to_datetime(amenity_details[['YEAR', 'MONTH']].assign(DAY=1))
            amenity_details = amenity_details[["Name","LATITUDE", "LONGITUDE", self.year_month_feature]]

        else:
            amenity_details = amenity_details[["address","LATITUDE", "LONGITUDE"]]

        no_of_amenities_within_radius = f"no_of_{amenity}_within_{radius}_km"
        distance_to_nearest_amenity = f"distance_to_nearest_{amenity}"

        if not self.inference_mode:
            columns = [no_of_amenities_within_radius, distance_to_nearest_amenity]
        else:
            columns = [no_of_amenities_within_radius, distance_to_nearest_amenity, f"nearest_{amenity}_coordinates", f"nearest_{amenity}_name"]

        amenity_features = pd.DataFrame(hdb_coordinates.progress_apply(lambda coordinates: hdb_est.utils.find_nearest_amenities(coordinates,
                                                                                                                               amenity_details = amenity_details,
                                                                                                                               radius = radius,
                                                                                                                               period = period,
                                                                                                                               latitude_feature = latitude_feature,
                                                                                                                               longitude_feature = longitude_feature,
                                                                                                                               year_month_feature = self.year_month_feature,
                                                                                                                               return_nearest_amenity = self.inference_mode),
                                                                                                                               axis = 1).tolist(),
                                                                                                                               columns=columns)
        
        return amenity_features


    


