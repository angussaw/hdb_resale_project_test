"""
## streamlit_app.py contains the front logic to load the dashboard that users
will interact with and input details of a hdb flat to generate predictions
"""

from datetime import datetime
import folium
from geopy.distance import geodesic
import hdb_resale_estimator as hdb_est
import json
import jsonpickle
import logging
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import requests
import shap
import streamlit as st
from streamlit_folium import folium_static
from typing import List

logger = logging.getLogger(__name__)


def validate_input_data(input: dict) -> List[str]:
    """Helper function to ensure input values can be used for inference

    Args:
        input (dict): Dictionary contain raw feature values for validation

    Returns:
        List[str]: list of messages informing what is wrong with the input data
    """

    messages = []

    if (
        datetime.strptime(input["month"], "%Y-%m").date().year
        < input["lease_commence_date"]
    ):
        messages.append("Lease commence year cannot be older than transaction year")

    if hdb_est.utils.find_coordinates(input["block"] + " " + input["street_name"]) == (
        float("inf"),
        float("inf"),
    ) | ((input["block"] == None) | (input["street_name"] == None)):
        messages.append("Please input a valid block and/or street name")

    if input["floor_area_sqm"] <= 0:
        messages.append("Please input a non-negative/zero value for the floor area parameter")

    if (input["lease_commence_date"] < 1900) | (input["lease_commence_date"] > 9999):
        messages.append("Please input a valid lease commence year")

    return messages


def main():
    """
    When button to estimate resale price is clicked:
        - Validation of input data
        - Display basic flat details on dashboard
        - Post request to backend to perform data prep
        - Display additional amenity information on dashboard
        - Post request to backend to predict resale value
        - Render map showing the hdb flat surroundings and nearby amenities
        - Post request to backend to generate shap values
        - Render waterfall plot of shap values to explain model prediction
    """

    hide_default_format = """
       <style>
       .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
    logger.info("Loading dashboard...")
    st.set_page_config(
        layout="wide",
        page_title="HDB Resale Price Estimator",
        page_icon=Image.open("images/house.png"),
    )
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title("HDB Resale Price Estimator")
    st.caption("This app estimates the resale price of a HDB flat given its details!")
    st.caption(
        "It also generates nearby amenities of the flat, and uses it to estimate the resale price"
    )
    st.caption(
        'Please input the flat details on the left sidebar and click "Estimate resale price" to generate predictions!'
    )
    with st.sidebar:
        submit = st.button("Estimate resale price")

        year = st.selectbox(
            "Select year", (["2015", "2016", "2017", "2018", "2019", "2020"])
        )

        month = st.selectbox(
            "Select month",
            (["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]),
        )

        month = year + "-" + month

        flat_type = st.radio(
            "Select flat type", ("3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE")
        )

        block = st.text_input("Block")

        street_name = st.text_input("Street Name")

        storey_range = st.selectbox(
            "Select storey range",
            (
                "01 TO 03",
                "04 TO 06",
                "07 TO 09",
                "10 TO 12",
                "13 TO 15",
                "16 TO 18",
                "19 TO 21",
                "22 TO 24",
                "25 TO 27",
                "28 TO 30",
                "31 TO 33",
                "34 TO 36",
                "37 TO 39",
                "40 TO 42",
                "43 TO 45",
                "46 TO 48",
                "49 TO 51",
            ),
        )

        floor_area_sqm = st.number_input("Floor area in square meters", step=0.01)

        lease_commence_year = st.number_input("Lease commence year", step=1, value=1966)

        town_name = st.selectbox(
            "Select town",
            (
                "ANG MO KIO",
                "BEDOK",
                "BISHAN",
                "BUKIT BATOK",
                "BUKIT MERAH",
                "BUKIT PANJANG",
                "BUKIT TIMAH",
                "CENTRAL AREA",
                "CHOA CHU KANG",
                "CLEMENTI",
                "GEYLANG",
                "HOUGANG",
                "JURONG EAST",
                "JURONG WEST",
                "KALLANG/WHAMPOA",
                "MARINE PARADE",
                "PASIR RIS",
                "PUNGGOL",
                "QUEENSTOWN",
                "SEMBAWANG",
                "SENGKANG",
                "SERANGOON",
                "TAMPINES",
                "TOA PAYOH",
                "WOODLANDS",
                "YISHUN",
            ),
        )

        flat_model_name = st.selectbox(
            "Select flat model",
            (
                "2-room",
                "Adjoined flat",
                "Apartment",
                "DBSS",
                "Improved",
                "Improved-Maisonette",
                "Maisonette",
                "Model A",
                "Model A-Maisonette",
                "Model A2",
                "Multi Generation",
                "New Generation",
                "Premium Apartment",
                "Premium Apartment Loft",
                "Premium Apartment.",
                "Premium Maisonette",
                "Simplified",
                "Standard",
                "Terrace",
                "Type S1",
                "Type S2",
            ),
        )

    if submit:
        input = {
            "id": 1,
            "month": month,
            "town": town_name,
            "flat_type": flat_type,
            "block": block.upper(),
            "street_name": street_name.upper(),
            "storey_range": storey_range,
            "floor_area_sqm": floor_area_sqm,
            "flat_model": flat_model_name,
            "lease_commence_date": lease_commence_year,
            "remaining_lease": "",
        }

        # Validation of input data
        messages = validate_input_data(input)

        if len(messages) > 0:
            for message in messages:
                st.write(message)

        else:
            # Display basic flat details on dashboard
            st.subheader("Flat details")
            input_df = pd.DataFrame([input], index=["Value"])
            input_df = input_df[
                [
                    col
                    for col in input_df.columns
                    if col not in ["id", "remaining_lease"]
                ]
            ]
            input_df = input_df.rename(columns={"month": "year-month"})
            st.table(data=input_df)

            # Post request to backend to perform data prep
            derived_input_data = requests.post(
                url="http://backend:8500/dataprep", data=json.dumps(input, default=str)
            )

            # Post request to backend to predict resale value
            predicted_resale_value = requests.post(
                url="http://backend:8500/predict",
                data=json.dumps(derived_input_data.json(), default=str),
            )

            # Display additional amenity information on dashboard
            derived_input_data_df = pd.DataFrame(
                [derived_input_data.json()], index=["Value"]
            )
            number_of_amenities_df = derived_input_data_df[
                [
                    col
                    for col in derived_input_data_df.columns
                    if col.startswith("no_of")
                ]
            ]
            radius = [int(x) for x in number_of_amenities_df.columns[0] if x.isdigit()][
                0
            ]
            number_of_amenities_df.columns = [
                col.replace("no_of_", "").replace(f"_within_{radius}_km", "").upper()
                for col in number_of_amenities_df.columns
            ]

            distance_to_nearest_amenity_df = derived_input_data_df[
                [
                    col
                    for col in derived_input_data_df.columns
                    if col.startswith("distance_to_nearest")
                ]
            ]
            distance_to_nearest_amenity_df.columns = [
                col.replace("distance_to_nearest_", "").upper()
                for col in distance_to_nearest_amenity_df.columns
            ]

            left, right = st.columns(2)
            with left:
                st.subheader(f"Number of nearby amenities within {radius} km")
                st.table(data=number_of_amenities_df)

            with right:
                st.subheader(f"Distances to nearest amenity (km)")
                st.table(data=distance_to_nearest_amenity_df)

            st.subheader("Predicted resale value:")
            st.write(round(float(predicted_resale_value.text), 0))

            # Render map showing the hdb flat surroundings and nearby amenities
            flat_coordinates = [
                derived_input_data_df.iloc[0]["latitude"],
                derived_input_data_df.iloc[0]["longitude"],
            ]
            flat_address = (
                derived_input_data_df.iloc[0]["block"]
                + " "
                + derived_input_data_df.iloc[0]["street_name"]
            )
            map = folium.Map(location=flat_coordinates, zoom_start=14)
            folium.Marker(
                location=flat_coordinates,
                popup=f"<b>{flat_address}/b>",
                tooltip=flat_address,
                icon=folium.Icon(color="blue", icon="home"),
            ).add_to(map)

            folium.Circle(flat_coordinates, radius=radius * 1000).add_to(map)

            icon_settings = {
                "MRT_stations": {"icon": "train", "color": "red"},
                "schools": {"icon": "user-graduate", "color": "orange"},
                "parks": {"icon": "tree", "color": "green"},
                "malls": {"icon": "store", "color": "purple"},
            }
            nearest_amenity_coordinates_col = derived_input_data_df.columns[
                derived_input_data_df.columns.str.endswith("coordinates")
            ]
            for col in nearest_amenity_coordinates_col:
                amenity = col.replace("nearest_", "").replace("_coordinates", "")
                amenity_name = derived_input_data_df.iloc[0][f"nearest_{amenity}_name"]
                amenity_coordinates = derived_input_data_df.iloc[0][col]

                distance = (
                    round(
                        float(
                            str(
                                geodesic(
                                    tuple(flat_coordinates), tuple(amenity_coordinates)
                                )
                            )[:-3]
                        ),
                        3,
                    )
                    * 1000
                )
                displacement_coordinates = [
                    tuple(flat_coordinates),
                    tuple(amenity_coordinates),
                ]
                folium.Marker(
                    location=amenity_coordinates,
                    popup=f"<b>{amenity_name}/<b>",
                    tooltip=amenity_name,
                    icon=folium.Icon(
                        color=icon_settings[amenity]["color"],
                        icon=icon_settings[amenity]["icon"],
                        prefix="fa",
                    ),
                ).add_to(map)

                folium.PolyLine(
                    displacement_coordinates, tooltip=f"{str(distance)} meters"
                ).add_to(map)

            left, right = st.columns(2)
            with left:
                st.subheader("Map of nearest amenities")
                folium_static(map, width=450, height=400)

            with right:
                # Post request to backend to generate shap values
                shap_values = requests.post(
                    url="http://backend:8500/explain",
                    data=json.dumps(derived_input_data.json(), default=str),
                )
                if shap_values.json():
                    # Render waterfall plot of shap values to explain model prediction
                    shap_values = jsonpickle.decode(shap_values.json())
                    st.subheader("SHAP values summary")
                    fig, ax = plt.subplots(nrows=1, ncols=1)
                    shap.plots.waterfall(shap_values, max_display=15, show=False)
                    st.pyplot(fig)

    else:
        pass


if __name__ == "__main__":
    main()
