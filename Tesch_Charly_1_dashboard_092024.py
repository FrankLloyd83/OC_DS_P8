import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import shap
from azure.storage.blob import BlobServiceClient
import os
import io

base_url = "https://oc-ds-p7-gqa9eqdze5hhakg3.eastus-01.azurewebsites.net"
predict_url = f"{base_url}/predict"
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING").replace('"', "")


@st.cache_data
def load_data(container_name, connection_string):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="DATA/application_train.csv")
    csv_data = blob_client.download_blob().readall()
    data = pd.read_csv(io.BytesIO(csv_data))

    blob_client = blob_service_client.get_blob_client(container=container_name, blob="DATA/data_prep/X_test.npy")
    npy_data = blob_client.download_blob().readall()
    clients_sample_data = np.load(io.BytesIO(npy_data))

    blob_client = blob_service_client.get_blob_client(container=container_name, blob="DATA/data_prep/features_names.txt")
    features_names_data = blob_client.download_blob().readall()
    features_names = features_names_data.decode("utf-8").replace("[", "").replace("]", "")[1:-1].split("', '")
    
    clients_sample_df = pd.DataFrame(clients_sample_data, columns=features_names)
    clients_sample_df["SK_ID_CURR"] = clients_sample_df["SK_ID_CURR"].astype(int)
    return data, clients_sample_df


def create_label_mappings(data):
    binary_categorical_columns = [
        col
        for col in data.select_dtypes(include=["object", "category"]).columns
        if data[col].nunique() == 2
    ]
    label_mappings = {
        col: dict(enumerate(data[col].dropna().unique()))
        for col in binary_categorical_columns
    }
    reverse_label_mappings = {
        col: {v: k for k, v in mapping.items()}
        for col, mapping in label_mappings.items()
    }
    return label_mappings, reverse_label_mappings


@st.cache_data
def get_prediction(features):
    try:
        response = requests.post(predict_url, json={"features": features})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error while requesting API: {e}")
        return None


def plot_probability_gauge(probability, threshold, delta=0):
    if probability < threshold:
        status_text = "Granted"
        status_color = "green"
    else:
        status_text = "Denied"
        status_color = "red"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability,
            gauge={
                "axis": {"range": [0, 1]},
                "threshold": {
                    "line": {"color": "orange", "width": 4},
                    "thickness": 0.75,
                    "value": threshold,
                },
                "bar": {"color": "midnightblue"},
                "steps": [
                    {"range": [0, threshold], "color": "#ADD8E6"},
                    {"range": [threshold, 1], "color": "#FFB6C1"},
                ],
            },
            number={
                "valueformat": ".0%",
                "suffix": f" ({delta:+.0%})" if delta else "",
            },
            title={"text": "Probability of Default"},
            delta={"reference": threshold, "valueformat": ".2f"},
        )
    )

    fig.add_annotation(
        x=0.5,
        y=0.4,
        text=status_text,
        showarrow=False,
        font={"size": 30, "color": status_color},
        xref="paper",
        yref="paper",
        align="center",
    )
    return fig


def init_session_state():
    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None
    if "shap_df" not in st.session_state:
        st.session_state.shap_df = None
    if "modified_features" not in st.session_state:
        st.session_state.modified_features = None


def plot_comparison(data, client_sample_df, variable, client_value):
    combined_df = pd.concat([data, client_sample_df], axis=0, ignore_index=True)
    try:
        lower_bound = combined_df[variable].quantile(0.01)
        upper_bound = combined_df[variable].quantile(0.99)
        within_bounds = combined_df[variable].between(lower_bound, upper_bound)
        outliers = combined_df[~within_bounds]
    except Exception as e:
        lower_bound, upper_bound = None, None
        outliers = pd.Series()

    fig, ax = plt.subplots()
    sns.histplot(
        combined_df[variable], label="All clients", color="blue", alpha=0.5, ax=ax
    )
    ax.axvline(client_value, color="red", label="Selected client", linestyle="--")
    ax.set_title(f"Comparison of {variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("Frequency")
    ax.legend()
    if not outliers.empty:
        ax.set_xlim(lower_bound, upper_bound)
    return fig


def plot_shap_waterfall(shap_df):
    expl = shap.Explanation(
        shap_df["SHAP Value"].values,
        feature_names=shap_df["Feature"].values,
        base_values=0,
    )
    fig, ax = plt.subplots()
    shap.plots.waterfall(expl, show=False)
    st.pyplot(fig)


@st.cache_data
def send_modified_features(modified_features, client_data):
    modified_client_data = {**client_data, **modified_features}
    return get_prediction(list(modified_client_data.values()))


def main():
    data, clients_sample_df = load_data(container_name="data", connection_string=connection_string)
    label_mappings, reverse_label_mappings = create_label_mappings(data)

    clients_sample_df.set_index("SK_ID_CURR", inplace=True)
    clients_sample_df_display = clients_sample_df.copy()

    for col, mapping in label_mappings.items():
        if col in clients_sample_df.columns:
            clients_sample_df_display[col] = clients_sample_df[col].map(mapping)

    col1, col2 = st.columns(2)
    with col1:
        client_id = st.selectbox("Select a client ID", clients_sample_df.index)
        client_data = clients_sample_df.loc[client_id].to_dict()

    with col2:
        variable_to_compare = st.selectbox(
            "Select a variable to compare", options=clients_sample_df.columns[:-1]
        )
    st.pyplot(
        plot_comparison(
            data,
            clients_sample_df_display,
            variable_to_compare,
            client_data[variable_to_compare],
        )
    )

    init_session_state()

    if st.button("Get prediction"):
        prediction_result = get_prediction(list(client_data.values()))
        if prediction_result:
            st.session_state.prediction_result = prediction_result
            st.session_state.shap_df = (
                pd.DataFrame(
                    {
                        "Feature": clients_sample_df.columns,
                        "SHAP Value": prediction_result["shap_values"],
                    }
                )
                .assign(abs_SHAP=lambda df: df["SHAP Value"].abs())
                .sort_values("abs_SHAP", ascending=False)
            )
            st.session_state.shap_df_neg = st.session_state.shap_df.sort_values(
                "SHAP Value", ascending=True
            ).head(5)
            st.session_state.shap_df_pos = st.session_state.shap_df.sort_values(
                "SHAP Value", ascending=False
            ).head(5)
            combined_features = (
                st.session_state.shap_df_neg["Feature"].tolist()
                + st.session_state.shap_df_pos["Feature"].tolist()
            )
            st.session_state.modified_features = {
                feature: client_data[feature] for feature in combined_features
            }

    if st.session_state.prediction_result:
        probability = st.session_state.prediction_result["proba"]
        threshold = st.session_state.prediction_result["threshold"]
        st.plotly_chart(plot_probability_gauge(probability, threshold))
        st.markdown(
            "<h4 style='text-align: center;'>Top 5 Negative SHAP Values:</h4>",
            unsafe_allow_html=True,
        )
        plot_shap_waterfall(st.session_state.shap_df_neg)

        st.markdown(
            "<h4 style='text-align: center;'>Top 5 Positive SHAP Values:</h4>",
            unsafe_allow_html=True,
        )
        plot_shap_waterfall(st.session_state.shap_df_pos)

        st.markdown(
            "<h4 style='text-align: center;'>Change the values of the most important features:</h4>",
            unsafe_allow_html=True,
        )

        modified_features = {}
        for feature in st.session_state.modified_features.keys():
            current_value = st.session_state.modified_features[feature]
            modified_value = st.slider(
                f"Modify {feature}",
                min_value=float(clients_sample_df[feature].min()),
                max_value=float(clients_sample_df[feature].max()),
                value=float(current_value),
            )
            st.session_state.modified_features[feature] = modified_value
            modified_features[feature] = modified_value

        if st.button("Send modifications"):
            modified_prediction = send_modified_features(modified_features, client_data)
            if modified_prediction:
                probability = modified_prediction["proba"]
                threshold = modified_prediction["threshold"]
                delta = probability - st.session_state.prediction_result["proba"]
                st.plotly_chart(plot_probability_gauge(probability, threshold, delta))


if __name__ == "__main__":
    main()
