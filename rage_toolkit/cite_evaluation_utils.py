import pandas as pd
from .types import PromptGenerator, LlmSystem, AugmentedGenerationSystem
import re
import random
import json

'''
==================
Section 1: Collection and Preprocessing of LLM Answers
==================
'''


def collect_query_response_info(query_row, augmented_generation_system: AugmentedGenerationSystem,
                                corpus_df: pd.DataFrame, relevant_df: pd.DataFrame, irrelevant_df: pd.DataFrame,
                                semirelevant_df: pd.DataFrame, proportions_mix):
    query_id: str = query_row['_id']
    query_text = query_row['text']
    doc_ids_for_query, rel_doc_ids_for_query, irrel_doc_ids_for_query, semirel_doc_ids_for_query, labels = create_mix(query_id,
                                                                                                              relevant_df,
                                                                                                              irrelevant_df,
                                                                                                              semirelevant_df,
                                                                                                              n_rel=
                                                                                                              proportions_mix[
                                                                                                                  0],
                                                                                                              n_irrel=
                                                                                                              proportions_mix[
                                                                                                                  1],
                                                                                                              n_semirel=
                                                                                                              proportions_mix[
                                                                                                                  2])
    docs_for_query_df = get_docs_for_query(doc_ids_for_query, corpus_df)
    docs_for_query = combine_doc_title_and_text(docs_for_query_df)
    prompt, response = augmented_generation_system.generate_answer(query_text, document_texts=docs_for_query)
    citations = parse_citations(response)
    query_response_info = {
        'query_id': query_id,
        'query_text': query_text,
        'prompt': prompt,
        'docs_for_query': docs_for_query,
        'answer': response,
        'answer_citations': citations,
        "doc_ids": doc_ids_for_query,
        "rel_doc_ids_for_query": rel_doc_ids_for_query,
        "irrel_doc_ids_for_query": irrel_doc_ids_for_query,
        "semirel_doc_ids_for_query": semirel_doc_ids_for_query
    }
    return query_response_info


def combine_doc_title_and_text(docs_for_query: pd.DataFrame):
    docs = []
    for i, doc_row in docs_for_query.iterrows():
        title = doc_row['title'] if type(doc_row['title']) is str else ""
        text = doc_row['text'] if type(doc_row['text']) is str else ""
        docs.append("\n".join([text, title]))
    return docs


def sample_docs_for_query(query_id: str, rel: pd.DataFrame, max_docs: int) -> list[str]:
    rel_filtered = rel[rel['query-id'] == query_id]
    size = rel_filtered.shape[0]
    n_sample = min(size, max_docs)
    rel_sampled = rel_filtered.sample(n=n_sample)
    return rel_sampled['corpus-id'].tolist()


def get_docs_for_query(doc_ids: list, corpus_df: pd.DataFrame):
    filtered_corpus_df = corpus_df[corpus_df['_id'].isin(doc_ids)]
    ordered_corpus_df = filtered_corpus_df.set_index('_id').loc[doc_ids].reset_index()
    return ordered_corpus_df


def parse_citations(answer: str):
    pattern = r'\[(\d+)\]'
    citations = re.findall(pattern, answer)
    return citations


def create_mix(query_id, relevant_df: pd.DataFrame, irrelevant_df: pd.DataFrame, semirelevant_df: pd.DataFrame,
               n_rel: int, n_irrel: int, n_semirel: int, randomize: bool = True):
    relevant_docs_for_query = sample_docs_for_query(query_id, relevant_df, n_rel)
    irrelevant_docs_for_query = sample_docs_for_query(query_id, irrelevant_df, n_irrel)
    semirelevant_docs_for_query = sample_docs_for_query(query_id, semirelevant_df, n_semirel)
    all_doc_ids_for_query = []
    all_doc_ids_for_query.extend(relevant_docs_for_query)
    all_doc_ids_for_query.extend(irrelevant_docs_for_query)
    all_doc_ids_for_query.extend(semirelevant_docs_for_query)
    labels = [True] * len(relevant_docs_for_query) + [False] * (len(irrelevant_docs_for_query)+ len(semirelevant_docs_for_query))

    if randomize:
        combined = list(zip(all_doc_ids_for_query, labels))
        random.shuffle(combined)
        all_doc_ids_for_query, labels = zip(*combined)
        all_doc_ids_for_query = list(all_doc_ids_for_query)
        labels = list(labels)
    return all_doc_ids_for_query, relevant_docs_for_query, irrelevant_docs_for_query, semirelevant_docs_for_query, labels


'''
==================
Section 2: Calculation of Evaluation Results
==================
'''


def calculate_evaluation_result_for_query(query_response_info: dict, relevant_df: pd.DataFrame):
    # ToDo check if query response dict has correct format
    query_id = query_response_info['query_id']
    doc_ids_for_query = query_response_info['doc_ids']
    rel_doc_ids_for_query = query_response_info['rel_doc_ids_for_query']
    answer_string = query_response_info['answer']
    citations = query_response_info['answer_citations']

    # ToDo try catch if not parseable
    citations = [int(citation) for citation in citations]
    citations, n_invalid_cites = __remove_invalid_citation(citations, doc_ids_for_query)
    cited_doc_ids = __get_cited_doc_ids(citations, doc_ids_for_query)
    distinct_cited_doc_ids = set(cited_doc_ids)
    num_cited_distinct = len(distinct_cited_doc_ids)

    num_passed_total = len(doc_ids_for_query)
    num_passed_relevant = len(rel_doc_ids_for_query)

    cited_relevant_doc_ids = __get_relevant_doc_ids_from_all_cited_distinct(distinct_cited_doc_ids,
                                                                            rel_doc_ids_for_query)
    num_cited_relevant = len(cited_relevant_doc_ids)

    cite_precision, cite_recall = __calculate_cite_precision_and_recall(num_cited_distinct, num_cited_relevant,
                                                                        num_passed_relevant)

    answer_length = __get_answer_length(answer_string=answer_string)

    eval_result = {
        'query_id': query_id,
        'num_passed_relevant': num_passed_relevant,
        'num_passed_total': num_passed_total,
        'num_cited_relevant': num_cited_relevant,
        'num_cited_distinct': num_cited_distinct,
        'cite_precision': cite_precision,
        'cite_recall': cite_recall,
        'answer_length': answer_length
    }

    if 'short_answers' in relevant_df.columns:
        eval_result['short-answer-exact-match'] = __contains_short_answer(query_response_info['answer'], query_id, cited_relevant_doc_ids, relevant_df)
    return eval_result


def __remove_invalid_citation(citations: list[int], doc_ids: list[str]) -> tuple[list[int], int]:
    cleaned_citations = [x for x in citations if 0 <= x - 1 < len(doc_ids)]
    return cleaned_citations, len(citations) - len(cleaned_citations)


def __get_cited_doc_ids(citations: list[int], doc_ids: list[str]) -> list[str]:
    cited_doc_ids = [doc_ids[i - 1] for i in citations]
    return cited_doc_ids


def __get_relevant_doc_ids_from_all_cited_distinct(distinct_citations: set, relevant: list[str]) -> list[str]:
    if len(distinct_citations) == 0:
        return []
    relevant_doc_ids = []
    for cited_doc in distinct_citations:
        if cited_doc in relevant:
            relevant_doc_ids.append(cited_doc)
    return relevant_doc_ids


def __contains_short_answer(llm_answer_string, query_id, relevant_doc_ids, relevant_df: pd.DataFrame):
    filtered_relevant_df = relevant_df.loc[(relevant_df['query-id'] == query_id) & (
            relevant_df['corpus-id'].isin(relevant_doc_ids) & (relevant_df['short_answers'].notna()))]
    if len(filtered_relevant_df) > 0:
        for i, row in filtered_relevant_df.iterrows():
            for short_answer_candidate in json.loads(row['short_answers']):
                print(f"{llm_answer_string} *in* {short_answer_candidate}?")
                if short_answer_candidate in llm_answer_string:
                    return 1
        return 0
    else:
        return None


def __calculate_cite_precision_and_recall(num_cited_distinct: int, relevant_cited_count: int,
                                          num_passed_relevant: int) -> tuple[float, float]:
    return relevant_cited_count / num_cited_distinct if num_cited_distinct > 0 else 0, relevant_cited_count / num_passed_relevant


def __get_answer_length(answer_string):
    answer_length = len(answer_string.split(" "))
    return answer_length