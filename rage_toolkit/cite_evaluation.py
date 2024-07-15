import pandas as pd
from .types import PromptGenerator, LlmSystem, AugmentedGenerationSystem
import os
from .cite_evaluation_utils import *
from datetime import datetime


def evaluate(
        augmented_generation_system: AugmentedGenerationSystem,
        queries_df: pd.DataFrame,
        corpus_df: pd.DataFrame,
        relevant_df: pd.DataFrame,
        irrelevant_df: pd.DataFrame,
        semirelevant_df: pd.DataFrame,
        proportions_mix: tuple,
        evaluation_results_save_path: str = None,
        checkpoint_interval: int = 10,
        num_queries: int = -1,
        run_id: str = ""
):
    # ToDo Check Inputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = str(timestamp)

    queries_response_information = []
    queries_evaluation_results = []

    queries_df.reset_index(drop=True, inplace=True)
    for index, query_row in queries_df.iterrows():
        # break loop if index is greater num_queries
        index = int(index)
        if num_queries != -1 and index >= num_queries:
            break

        query_response_info = collect_query_response_info(
            query_row=query_row,
            augmented_generation_system=augmented_generation_system,
            corpus_df=corpus_df,
            relevant_df=relevant_df,
            irrelevant_df=irrelevant_df,
            semirelevant_df=semirelevant_df,
            proportions_mix=proportions_mix
        )
        queries_response_information.append(query_response_info)

        eval_result = calculate_evaluation_result_for_query(
            query_response_info=query_response_info,
            relevant_df=relevant_df
        )
        queries_evaluation_results.append(eval_result)

        if evaluation_results_save_path is not None and index % checkpoint_interval == 0:
            __save_query_info_and_eval_results(
                queries_response_information_df=pd.DataFrame(queries_response_information),
                queries_evaluation_results_df=pd.DataFrame(queries_evaluation_results),
                evaluation_results_save_path=evaluation_results_save_path,
                timestamp=timestamp,
                run_id=run_id)

        print(f"processed {index} of {queries_df.shape[0]} queries ({(index / queries_df.shape[0]) * 100}%)")

    queries_response_information_df = pd.DataFrame(queries_response_information)
    queries_evaluation_results_df = pd.DataFrame(queries_evaluation_results)

    if evaluation_results_save_path is not None:
        __save_query_info_and_eval_results(
            queries_response_information_df=pd.DataFrame(queries_response_information),
            queries_evaluation_results_df=pd.DataFrame(queries_evaluation_results),
            evaluation_results_save_path=evaluation_results_save_path,
            timestamp=timestamp,
            run_id=run_id)
    return queries_response_information_df, queries_evaluation_results_df


def __save_query_info_and_eval_results(queries_response_information_df, queries_evaluation_results_df,
                                       evaluation_results_save_path, timestamp:str, run_id: str):
    if not os.path.exists(os.path.dirname(evaluation_results_save_path)):
        os.makedirs(os.path.dirname(evaluation_results_save_path))
    queries_response_information_df.to_csv(os.path.join(evaluation_results_save_path, f'{timestamp}_response_info_{run_id}.csv'), index=False)
    queries_evaluation_results_df.to_csv(os.path.join(evaluation_results_save_path, f'{timestamp}_evaluation_results_{run_id}.csv'), index=False)
