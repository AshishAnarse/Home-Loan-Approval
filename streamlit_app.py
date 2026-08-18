# ============================================================
# STREAMLIT APPLICATION
# Machine Learning Assignment - 2
# Loan Sanction Model Evaluation
# ============================================================


# ------------------------------------------------------------
# 1. IMPORT LIBRARIES
# ------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------
# 2. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Loan Sanction ML",
    page_icon="🏦",
    layout="wide"
)


# ------------------------------------------------------------
# 3. APPLICATION TITLE
# ------------------------------------------------------------

st.title(
    "🏦 Loan Sanction Classification"
)

st.subheader(
    "Machine Learning Assignment - Model Evaluation Application"
)

st.write(
    """
    This application evaluates multiple machine-learning
    classification models for predicting loan approval status.

    Upload the supplied test_data.csv file, select a model,
    and view its classification performance.
    """
)


# ------------------------------------------------------------
# 4. AVAILABLE MODELS
# ------------------------------------------------------------

model_files = {

    "Logistic Regression":
        "model/logistic_regression.pkl",

    "Decision Tree":
        "model/decision_tree.pkl",

    "kNN":
        "model/knn.pkl",

    "Naive Bayes":
        "model/naive_bayes.pkl",

    "Random Forest":
        "model/random_forest.pkl"
}


# ------------------------------------------------------------
# 5. MODEL SELECTION DROPDOWN
# ------------------------------------------------------------

st.sidebar.header(
    "Model Configuration"
)


selected_model_name = st.sidebar.selectbox(
    "Select Machine Learning Model",
    list(model_files.keys())
)


st.sidebar.write(
    "Selected model:",
    selected_model_name
)


# ------------------------------------------------------------
# 6. CSV FILE UPLOAD
# ------------------------------------------------------------

st.header(
    "1. Upload Test Dataset"
)


uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ------------------------------------------------------------
# 7. PROCESS UPLOADED FILE
# ------------------------------------------------------------

if uploaded_file is not None:

    try:

        test_df = pd.read_csv(
            uploaded_file
        )


        # ----------------------------------------------------
        # Display uploaded data
        # ----------------------------------------------------

        st.success(
            "Test dataset uploaded successfully."
        )

        st.write(
            "Dataset Shape:",
            test_df.shape
        )


        st.subheader(
            "Uploaded Dataset Preview"
        )

        st.dataframe(
            test_df.head(10),
            use_container_width=True
        )


        # ----------------------------------------------------
        # Check target variable
        # ----------------------------------------------------

        if "Loan_Status" not in test_df.columns:

            st.error(
                """
                The uploaded test file must contain the
                Loan_Status column so that evaluation metrics
                can be calculated.
                """
            )

            st.stop()


        # ----------------------------------------------------
        # Prepare target
        # ----------------------------------------------------

        y_test = test_df[
            "Loan_Status"
        ].map(
            {
                "N": 0,
                "Y": 1
            }
        )


        # ----------------------------------------------------
        # Remove target from predictors
        # ----------------------------------------------------

        X_test = test_df.drop(
            columns=[
                "Loan_Status"
            ]
        )


        # ----------------------------------------------------
        # Re-create engineered features if necessary
        # ----------------------------------------------------

        if "TotalIncome" not in X_test.columns:

            X_test[
                "TotalIncome"
            ] = (
                X_test[
                    "ApplicantIncome"
                ] +
                X_test[
                    "CoapplicantIncome"
                ]
            )


        if (
            "LoanAmount_to_TotalIncome"
            not in X_test.columns
        ):

            X_test[
                "LoanAmount_to_TotalIncome"
            ] = (
                X_test[
                    "LoanAmount"
                ] /
                X_test[
                    "TotalIncome"
                ].replace(
                    0,
                    np.nan
                )
            )


        # ----------------------------------------------------
        # Remove Loan_ID if present
        # ----------------------------------------------------

        if "Loan_ID" in X_test.columns:

            X_test = X_test.drop(
                columns=[
                    "Loan_ID"
                ]
            )


        # ----------------------------------------------------
        # Load selected model
        # ----------------------------------------------------

        selected_model_path = model_files[
            selected_model_name
        ]


        model = joblib.load(
            selected_model_path
        )


        # ----------------------------------------------------
        # Prediction button
        # ----------------------------------------------------

        if st.button(
            "Evaluate Selected Model",
            type="primary"
        ):

            # ------------------------------------------------
            # Generate predictions
            # ------------------------------------------------

            y_pred = model.predict(
                X_test
            )


            # Probability of approved class
            y_probability = model.predict_proba(
                X_test
            )[:, 1]


            # ------------------------------------------------
            # Calculate required metrics
            # ------------------------------------------------

            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            auc = roc_auc_score(
                y_test,
                y_probability
            )

            precision = precision_score(
                y_test,
                y_pred,
                zero_division=0
            )

            recall = recall_score(
                y_test,
                y_pred,
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                y_pred,
                zero_division=0
            )

            mcc = matthews_corrcoef(
                y_test,
                y_pred
            )


            # ------------------------------------------------
            # Display selected model
            # ------------------------------------------------

            st.header(
                f"2. Results: {selected_model_name}"
            )


            # ------------------------------------------------
            # Display required metrics
            # ------------------------------------------------

            st.subheader(
                "Evaluation Metrics"
            )


            col1, col2, col3 = st.columns(
                3
            )


            with col1:

                st.metric(
                    "Accuracy",
                    f"{accuracy:.4f}"
                )


            with col2:

                st.metric(
                    "AUC",
                    f"{auc:.4f}"
                )


            with col3:

                st.metric(
                    "Precision",
                    f"{precision:.4f}"
                )


            col4, col5, col6 = st.columns(
                3
            )


            with col4:

                st.metric(
                    "Recall",
                    f"{recall:.4f}"
                )


            with col5:

                st.metric(
                    "F1 Score",
                    f"{f1:.4f}"
                )


            with col6:

                st.metric(
                    "MCC",
                    f"{mcc:.4f}"
                )


            # ------------------------------------------------
            # Confusion Matrix
            # ------------------------------------------------

            st.subheader(
                "Confusion Matrix"
            )


            cm = confusion_matrix(
                y_test,
                y_pred
            )


            fig, ax = plt.subplots(
                figsize=(5, 4)
            )


            ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=[
                    "Rejected",
                    "Approved"
                ]
            ).plot(
                ax=ax
            )


            ax.set_title(
                f"Confusion Matrix - {selected_model_name}"
            )


            st.pyplot(
                fig
            )


            # ------------------------------------------------
            # Classification Report
            # ------------------------------------------------

            st.subheader(
                "Classification Report"
            )


            report = classification_report(
                y_test,
                y_pred,
                target_names=[
                    "Rejected",
                    "Approved"
                ],
                output_dict=True,
                zero_division=0
            )


            report_df = pd.DataFrame(
                report
            ).transpose()


            st.dataframe(
                report_df.round(4),
                use_container_width=True
            )


            # ------------------------------------------------
            # Prediction Results
            # ------------------------------------------------

            st.subheader(
                "Prediction Results"
            )


            prediction_output = test_df.copy()


            prediction_output[
                "Predicted_Loan_Status"
            ] = np.where(
                y_pred == 1,
                "Y",
                "N"
            )


            prediction_output[
                "Approval_Probability"
            ] = y_probability.round(
                4
            )


            prediction_output[
                "Prediction_Correct"
            ] = (
                prediction_output[
                    "Loan_Status"
                ] ==
                prediction_output[
                    "Predicted_Loan_Status"
                ]
            )


            st.dataframe(
                prediction_output,
                use_container_width=True
            )


            # ------------------------------------------------
            # Optional CSV download
            # ------------------------------------------------

            output_csv = prediction_output.to_csv(
                index=False
            ).encode(
                "utf-8"
            )


            st.download_button(
                label="Download Prediction Results",
                data=output_csv,
                file_name=(
                    "loan_prediction_results.csv"
                ),
                mime="text/csv"
            )


    except Exception as error:

        st.error(
            f"An error occurred: {error}"
        )


# ------------------------------------------------------------
# 8. INITIAL MESSAGE WHEN NO FILE IS UPLOADED
# ------------------------------------------------------------

else:

    st.info(
        """
        Upload the generated test_data.csv file to begin
        model evaluation.
        """
    )


# ------------------------------------------------------------
# 9. FOOTER / MODEL INFORMATION
# ------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.write(
    """
    **Classification Models**

    - Logistic Regression
    - Decision Tree
    - kNN
    - Gaussian Naive Bayes
    - Random Forest
    """
)