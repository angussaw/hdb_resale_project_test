"""Module containing the Evaluator class with methods to evaluate a model's performance
"""
import logging
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate
from typing import Tuple

import hdb_resale_estimator as hdb_est
from hdb_resale_estimator.modeling.builder import ClassicalModelBuilder

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator class will calculate model performance metrics and
    generate visualizations for the train, validation and test sets

    Args:
        builder (ClassicalModelBuilder): Object containing the trained model
        params (dict): Dictionary containing the parameters for the evaluation
        chosen_model (str): Name of the trained model
    """

    def __init__(
        self,
        builder: ClassicalModelBuilder,
        params: dict,
        chosen_model: str,
    ) -> None:
        self.builder = builder
        self.top_n_features = params["feature_importance_top_n"]
        self.chosen_model = chosen_model
        self.rounding_last_n = params["rounding_last_n"]
        self.no_of_cv_folds = params["no_of_cv_folds"]
        self.shap_explainer = params["shap_explainer"]

    def evaluate_model(
        self,
        datasets: dict,
    ) -> Tuple[dict, str]:
        """
        Evaluates the model on specific datasets

        Args:
            datasets (dict): Dictionary containing the different datasets

        Returns:
            dict: Dictionary containing the model performance metrics for each dataset
            str: Directory that the evaluation visualizations are saved in
        """

        metrics = {}
        # generate tmp directory to save visualisation
        visualizations_save_dir = hdb_est.utils.generate_named_tmp_dir(dir_name="graph")

        with hdb_est.utils.timer(task="train-test-val model evaluation"):
            for dataset_type in datasets:
                logger.info("Evaluating %s set...", dataset_type)
                features = datasets[dataset_type]["X"]
                target = datasets[dataset_type]["y"]
                predicted_values = hdb_est.modeling.model.make_predictions(
                    builder=self.builder, data=features
                )

                logger.info("Calculating %s metrics...", dataset_type)
                dataset_metrics = self._calculate_metrics(
                    dataset_type=dataset_type,
                    actual_values=target,
                    predicted_values=predicted_values,
                    rounding_last_n=self.rounding_last_n,
                )
                metrics.update(dataset_metrics)

                self._save_actual_predicted_scatterplot(
                    dataset_type=dataset_type,
                    actual_values=target,
                    predicted_values=predicted_values,
                    save_dir=visualizations_save_dir,
                )

        # Generate cross validation scores
        if self.no_of_cv_folds:
            metrics.update(
                self._cross_val_scores(
                    X=datasets["train"]["X"],
                    y=datasets["train"]["y"],
                )
            )

        if self.chosen_model == "ebm":
            self._save_ebm_feature_importances(save_dir=visualizations_save_dir)

        elif self.chosen_model == "randforest":
            self._save_tree_feature_importances(
                train_X=datasets["train"]["X"], save_dir=visualizations_save_dir
            )

            if self.shap_explainer:
                self._generate_shap_plots(
                    train_X=datasets["train"]["X"], save_dir=visualizations_save_dir
                )

        elif (self.chosen_model == "xgboost") & (self.shap_explainer):
            self._generate_shap_plots(
                train_X=datasets["train"]["X"], save_dir=visualizations_save_dir
            )

        else:
            logger.info("%s not in list to show feature importance", self.chosen_model)

        return metrics, visualizations_save_dir

    def _calculate_metrics(
        self,
        dataset_type: str,
        actual_values: npt.ArrayLike,
        predicted_values: npt.ArrayLike,
        rounding_last_n: int,
    ) -> dict:
        """
        Calculates the relevant model performance metrics for a specific dataset

        Args:
            dataset_type (str): Train, validation or test set
            actual_values (npt.ArrayLike): Array containing actual values of the target label
            predicted_values (npt.ArrayLike): Array containing predicted values of the target label
            rounding_last_n (int): Last n digits for rounding off the metrics

        Returns:
            dict: Dictionary containing the following model performance metrics:
                - root_mean_squared_error
                - mean_absolute_error
                - r2_score


        """
        if not rounding_last_n:
            rounding_last_n = 5

        metrics = {
            f"{dataset_type}_root_mean_squared_error": round(
                np.sqrt(mean_squared_error(actual_values, predicted_values)),
                rounding_last_n,
            ),
            f"{dataset_type}_mean_absolute_error": round(
                mean_absolute_error(actual_values, predicted_values), rounding_last_n
            ),
            f"{dataset_type}_r2_score": round(
                r2_score(actual_values, predicted_values), rounding_last_n
            ),
        }

        return metrics

    def _generate_visualizations(
        self,
        dataset_type: str,
        actual_values: npt.ArrayLike,
        predicted_values: npt.ArrayLike,
        save_dir: str,
    ) -> dict:
        """_summary_

        Args:
            dataset_type (str): Train, validation or test set
            actual_values (npt.ArrayLike): Array containing actual values of the target label
            predicted_values (npt.ArrayLike): Array containing predicted values of the target label
            save_dir (str): Directory that the evaluation visualizations are saved in

        Returns:
            dict: Dictionary containing the save path of the following visualizations:
                    - Actual vs Predicted scatterplot
        """

        actual_predicted_scatterplot_path = self._save_actual_predicted_scatterplot(
            dataset_type=dataset_type,
            actual_values=actual_values,
            predicted_values=predicted_values,
            save_dir=save_dir,
        )

        visualizations = {
            "actual_predicted_scatterplot": actual_predicted_scatterplot_path
        }

        return visualizations

    def _save_actual_predicted_scatterplot(
        self,
        dataset_type: str,
        actual_values: npt.ArrayLike,
        predicted_values: npt.ArrayLike,
        save_dir: str,
    ) -> str:
        """
        Generates the actual vs predicted values scatterplot and saves the visualisation to filepath

        Args:
            dataset_type (str): Train, validation or test set
            actual_values (npt.ArrayLike): Array containing actual values of the target label
            predicted_values (npt.ArrayLike): Array containing predicted values of the target label
            save_dir (str): Directory that the evaluation visualizations are saved in

        Returns:
            str: File save path of the visualization
        """
        colors = actual_values - predicted_values
        plt.figure(figsize=(10, 10))
        plt.scatter(x=predicted_values, y=actual_values, c=colors, alpha=0.5)

        plt.xlabel("Predicted")
        plt.ylabel("Actual")

        plot_name = f"Actual_vs_Predicted_scatterplot ({dataset_type})"
        plt.title(plot_name)

        file_save_path = self._save_visualization(
            plot_name=plot_name, save_dir=save_dir
        )

        return file_save_path

    def _save_ebm_feature_importances(self, save_dir: str) -> str:
        """
        Generates a horizontal barplot of the ebm global feature importances and saves the visualisation to filepath

        Args:
            save_dir (str): Directory that the evaluation visualizations are saved in

        Returns:
            str: File save path of the visualization
        """
        explain_global = self.builder.model.explain_global()
        feature_impt = pd.DataFrame(explain_global.data())
        feature_impt["feature_types"] = explain_global.feature_types
        feature_impt = feature_impt[feature_impt["feature_types"] != "interaction"]
        feature_impt = feature_impt.sort_values(by=["scores"]).reset_index(drop=True)
        if self.top_n_features:
            feature_impt = feature_impt.tail(self.top_n_features)

        plt.figure(figsize=(15, 10))
        plt.barh(
            y=feature_impt["names"],
            width=feature_impt["scores"],
        )
        plt.xlabel("Mean Absolute Score")
        plot_name = "Overall_EBM_Feature_Importances"
        if self.top_n_features:
            plot_name = f"{plot_name}_top_{self.top_n_features}"
        plt.title(plot_name)

        file_save_path = self._save_visualization(
            plot_name=plot_name, save_dir=save_dir
        )

        return file_save_path

    def _save_tree_feature_importances(self, train_X: pd.DataFrame, save_dir: str):
        """
        Generates a horizontal barplot of the feature importances in a tree-based model
        and saves the visualisation to filepath

        Args:
            train_X (pd.DataFrame): Dataframe containing train set features
            save_dir (str): Directory that the evaluation visualizations are saved in

        Returns:
            str: File save path of the visualization
        """

        feature_impt_df = pd.DataFrame()
        feature_impt_df["Feature"] = list(train_X.columns)
        feature_impt_df["Importance"] = list(self.builder.model.feature_importances_)
        feature_impt_df = feature_impt_df.sort_values(
            by=["Importance"], ascending=False
        )
        feature_impt_df = feature_impt_df.head(self.top_n_features)

        plt.figure(figsize=(15, 10))
        plt.barh(
            y=feature_impt_df["Feature"],
            width=feature_impt_df["Importance"],
        )
        plt.xlabel("Importance")
        plot_name = f"Overall_{self.chosen_model}_Feature_Importances"
        if self.top_n_features:
            plot_name = f"{plot_name}_top_{self.top_n_features}"
        plt.title(plot_name)

        file_save_path = self._save_visualization(
            plot_name=plot_name, save_dir=save_dir
        )

        return file_save_path

    def _save_visualization(self, plot_name: str, save_dir: str) -> str:
        """
        Helper function to save visualization to specified directory

        Args:
            plot_name (str): Name of the visualization
            save_dir (str): Directory that the evaluation visualizations are saved in

        Returns:
            str: File save path of the visualization
        """

        file_name = f"{plot_name}.png"
        file_save_path = f"{save_dir}/{file_name}"
        plt.savefig(file_save_path, bbox_inches="tight")

        return file_save_path

    def _cross_val_scores(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Helper function to calculate cross validation scores

        Args:
            X (pd.DataFrame): Dataframe containing features
            y (pd.Series): Target labels

        Returns:
            dict: Dictionary of cross validation scores
        """
        # Clone model and get an unfitted model
        clone_model = clone(self.builder.model)
        scorings = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        result = cross_validate(
            clone_model,
            X,
            y,
            scoring=scorings,
            cv=self.no_of_cv_folds,
        )
        metrics = {
            "cv_mean_root_mean_squared_error": np.mean(
                [np.sqrt(-mse) for mse in result["test_neg_mean_squared_error"]]
            ),
            "cv_std_root_mean_squared_error": np.std(
                [np.sqrt(-mse) for mse in result["test_neg_mean_squared_error"]]
            ),
            "cv_mean_mean_absolute_error": -result[
                "test_neg_mean_absolute_error"
            ].mean(),
            "cv_std_mean_absolute_error": result["test_neg_mean_absolute_error"].std(),
            "cv_mean_r2_score": result["test_r2"].mean(),
            "cv_std_r2_score": result["test_r2"].std(),
        }
        return metrics

    def _generate_shap_plots(self, train_X: pd.DataFrame, save_dir: str):
        """
        Generates shap waterfall plot using calculated shap values of the train set
        and saves the visualisation to filepath

        Args:
            train_X (pd.DataFrame): Dataframe containing train set features
            save_dir (str): Directory that the evaluation visualizations are saved in

        Returns:
            str: File save path of the visualization
        """

        explainer = shap.Explainer(self.builder.model.predict, train_X)
        self.builder.objects["explainer"] = explainer

        shap_values = explainer(train_X)

        shap.summary_plot(
            shap_values.values,
            features=train_X,
            feature_names=list(train_X.columns),
            show=False,
        )
        _, h = plt.gcf().get_size_inches()
        plt.gcf().set_size_inches(h * 3, h)
        ax = plt.gca()
        ax.set_xlim(-800000, 800000)
        plot_name = f"Summary plot of shap values"
        plt.title(plot_name)

        file_save_path = self._save_visualization(
            plot_name=plot_name, save_dir=save_dir
        )

        return file_save_path
