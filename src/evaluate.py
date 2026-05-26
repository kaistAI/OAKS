from email.policy import default
import json
import re
import csv
from collections import defaultdict
from typing import Union, List
import numpy as np
from tqdm import tqdm
import os

    
def read_jsonl(file_path: Union[str, List[str]], qids_to_answer: dict, is_novel: bool):
    data = []
    data_filtered = []
    if isinstance(file_path, str):
        file_path = [file_path]

    q_filter_ids = {}
    for path in file_path:
        data_ = []
        keys_ = {}
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                qid = item.get('doc_id') or item.get('query_id')
                if 'doc_id' not in item:
                    item['doc_id'] = item['query_id']
                    
                data_.append(item)
                keys_[qid] = qid
                question_text = item.get('query').strip()
                if "Current Head Index" in question_text:
                    question_text = question_text.split("question:")[-1].strip()
                
                q_filter_id = qid if is_novel and not qid.startswith("C12") else "_".join(qid.split("_")[:-1]) + "_" + question_text
                
                if q_filter_id in qids_to_answer:
                    item["q_filter_id"] = q_filter_id
                    if q_filter_id not in q_filter_ids:
                        q_filter_ids[q_filter_id] = qid
                        data_filtered.append(item)
        data.extend(data_)
    
    return data_filtered, len(data)

    
def extract_single_char_answer(prediction: str):
    match = re.search(r'\b[A-Z]\b', prediction)
    if match:
        return match.group(0)
    return ""
    
def extract_answer(prediction: str, novel: bool = False):
    # Updated regex to handle:
    # 1. Variations of "Answer:": "answer":, **answer**:, ### Answer:, etc.
    # 2. "The answer is", "The final answer is"
    # pattern = r'(?:(?:["\'*#]*|###\s+)answer["\'*#]*\s*:|(?:final\s+)?answer\s+is[:\s]*)'
    
    pattern = r'(?:(?:["\'*#]*|###\s+)answer["\'*#]*\s*:)'
    
    parts = re.split(pattern, prediction, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[-1].strip()
        
    if novel:
        return ''
    else: 
        return prediction
    # return ''

def all_golds():
    import json
    with open("/mnt/nas/jiyeon/ltmcl/data_processed/babilong/final/filtered.total_qa_set.json", "r") as f:
        data = json.load(f)
    gts = []
    for factset in data:
        for qa_type, qas in factset['replaced_qas'].items():
            for qa in qas:
                gts.extend(list(qa['answer'].values()))
    print(f"Number of gts: {len(gts)}")
    print(f"Number of unique gts: {len(set(gts))}")
    return set(gts)

    
    
def is_correct_babi(answer, gold):
    int_to_text = {
        0: ['zero', 'never'],
        1: ['one', 'once'],
        2: ['two', 'twice'],
        3: ['three', 'thrice', 'three times'],
        4: ['four', 'four times'],
        5: ['five', 'five times'],
        6: ['six', 'six times'],
        7: ['seven', 'seven times'],
        8: ['eight', 'eight times'],
        9: ['nine', 'nine times'],
        10: ['ten', 'ten times'],
        11: ['eleven', 'eleven times'],
        12: ['twelve', 'twelve times'],
        13: ['thirteen', 'thirteen times'],
        14: ['fourteen', 'fourteen times'],
        15: ['fifteen', 'fifteen times'],
        16: ['sixteen', 'sixteen times'],
        17: ['seventeen', 'seventeen times'],
        18: ['eighteen', 'eighteen times'],
        19: ['nineteen', 'nineteen times'],
        20: ['twenty', 'twenty times'],
        21: ['twenty-one', 'twenty-one times'],
        22: ['twenty-two', 'twenty-two times'],
        23: ['twenty-three', 'twenty-three times'],
    }
    if answer.lower() == 'unknown' and gold in ['NA', '0', 0, 'same']:
        return True
    elif isinstance(gold, int) and str(gold) == answer:
        return True
    elif str(gold).isdigit() and int(gold) in int_to_text:
        int_to_text[int(gold)] += [str(gold)]
        return answer.lower() in int_to_text[int(gold)]
    else:
        return answer.lower() == str(gold).lower()

def _normalize_answer(s, is_MCQ: bool = False):
    import string
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        if isinstance(text, int):
            return str(text).lower()
        return text.lower()
    
    if is_MCQ:
        return white_space_fix(remove_punc(lower(s)))
    else:
        return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(answer, gold, is_MCQ: bool = False):
    assert isinstance(gold, list)
    assert isinstance(answer, str)
    answer = _normalize_answer(answer, is_MCQ)
    gold = [_normalize_answer(g, is_MCQ) for g in gold]
    
    return answer in set(gold)

def data_q_to_chunk(data, novel, qids_to_answer, qids_to_type):
    q_to_chunk = defaultdict(lambda: defaultdict(lambda: {"correct": False, "GT": "", "answer": "", "prediction": ""}))
    
    qid_to_type = {}
    for item in data:
        prediction = item['generated_answer']
        answer = extract_answer(prediction, novel)
        
        if novel:
            answer = extract_single_char_answer(answer)
        
        q_filter_id = item.get('q_filter_id')
        bid, chunk_idx, question_idx = q_filter_id.split("_")
        
        qid = bid +"_" + question_idx
        gold_list = qids_to_answer[q_filter_id]
        is_correct = exact_match_score(answer, gold_list, is_MCQ=novel) if answer != '' else False
        
        qtype = qids_to_type[q_filter_id]
        qid_to_type[qid] = qtype
        
        q_to_chunk[qid][int(chunk_idx)+1]["correct"] = is_correct
        q_to_chunk[qid][int(chunk_idx)+1]["GT"] = gold_list[0]
        q_to_chunk[qid][int(chunk_idx)+1]["answer"] = answer
        q_to_chunk[qid][int(chunk_idx)+1]["prediction"] = prediction
        q_to_chunk[qid][int(chunk_idx)+1]["query"] = item['query']
    q_to_chunk_sorted = {}
    for qid in q_to_chunk.keys():
        q_to_chunk_sorted[qid] = dict(sorted(q_to_chunk[qid].items(), key=lambda x: int(x[0])))
    
    return q_to_chunk_sorted, qid_to_type
        
        
def calculate_acc_score(q_to_chunk_babi):
    
    q_to_score_avg = {}
    q_to_score = {}
    for qid, chunk_result in q_to_chunk_babi.items():
        acc_score = 0
        N = len(chunk_result)
        for chunk_idx, chunk_data in chunk_result.items():
            k_t = chunk_idx / N
            acc_score += int(chunk_data['correct']) * k_t
        acc_score *= (2/(N+1))
        q_to_score[qid] = acc_score
        q_to_score_avg[qid] = sum([int(chunk_data['correct']) for chunk_data in chunk_result.values()])/N

    final_acc_score = list(q_to_score.values())
    simple_avg_acc_score = list(q_to_score_avg.values())
    
    
    return sum(final_acc_score)/len(final_acc_score), sum(simple_avg_acc_score)/len(simple_avg_acc_score)


def calculate_simple_score(q_to_chunk_babi, t=None):
    
    q_to_state_stats = defaultdict(lambda: defaultdict(
            lambda:  {'durations': [], 'latencies': [], 'distractions': [], 'correct_num': []}
        ))
    for qid, chunks in q_to_chunk_babi.items():
        sorted_chunks = sorted(chunks.items(), key=lambda x: int(x[0]))
        if not sorted_chunks: continue
            
        current_gt = sorted_chunks[0][1]['GT']
        current_segment = []
        # import pdb; pdb.set_trace()
        for chunk_idx, data in sorted_chunks:
            gt = data['GT']
            
            # If GT changes, process the completed segment
            if gt != current_gt:
                duration = len(current_segment)
                
                # Find first correct index (latency)
                # If not found, default to duration as requested
                first_correct_idx = next((i for i, item in enumerate(current_segment) if item['correct']), duration+1)
                latency = first_correct_idx if first_correct_idx <= duration else duration
                correct_num = sum([1 if item['correct'] else 0 for item in current_segment])
                distraction = sum([1 if i > first_correct_idx and not item['correct'] else 0 for i, item in enumerate(current_segment)])
                q_to_state_stats[qid][current_gt]['durations'].append(duration)
                q_to_state_stats[qid][current_gt]['latencies'].append(latency)
                q_to_state_stats[qid][current_gt]['distractions'].append(distraction)
                q_to_state_stats[qid][current_gt]['correct_num'].append(correct_num)

                current_gt = gt
                current_segment = []
            
            current_segment.append(data)
            
        # Process the final segment
        if current_segment:
            duration = len(current_segment)
            first_correct_idx = next((i for i, item in enumerate(current_segment) if item['correct']), duration+1)
            latency = first_correct_idx if first_correct_idx <= duration else duration
            correct_num = sum([1 if item['correct'] else 0 for item in current_segment])
            distraction = sum([1 if i > first_correct_idx and not item['correct'] else 0 for i, item in enumerate(current_segment)])
            q_to_state_stats[qid][current_gt]['durations'].append(duration)
            q_to_state_stats[qid][current_gt]['latencies'].append(latency)
            q_to_state_stats[qid][current_gt]['distractions'].append(distraction)
            q_to_state_stats[qid][current_gt]['correct_num'].append(correct_num)

    q_to_acq_score = {}
    q_to_robust_score = {}
    q_to_acq_lenient_score = {}
    q_to_ds_fixed_score = {}
    
    for qid, state_stats in q_to_state_stats.items():
        latency_scores = []
        latency_lenient_scores = []
        distraction_scores = []
        distraction_fixed_scores = []
        for gt, stats in state_stats.items():
            for duration, latency, distraction, correct_num in zip(stats['durations'], stats['latencies'], stats['distractions'], stats['correct_num']):
                latency_scores.append(latency/duration)
                distraction_scores.append(distraction/duration)
                
                latency_lenient_scores.append(latency/duration if latency < duration else 0)
                distraction_fixed_scores.append(distraction/duration if correct_num != 0 else 1)

        
        q_to_acq_score[qid] = sum(latency_scores)/len(latency_scores)
        q_to_robust_score[qid] = sum(distraction_scores)/len(distraction_scores)
        
        q_to_acq_lenient_score[qid] = sum(latency_lenient_scores)/len(latency_lenient_scores)
        q_to_ds_fixed_score[qid] = sum(distraction_fixed_scores)/len(distraction_fixed_scores)

    final_acq_score = list(q_to_acq_score.values())
    AL = sum(final_acq_score)/len(final_acq_score)
    final_robust_score = list(q_to_robust_score.values())
    DS = sum(final_robust_score)/len(final_robust_score)
    final_acq_lenient_score = list(q_to_acq_lenient_score.values())
    AL_lenient = sum(final_acq_lenient_score)/len(final_acq_lenient_score)
    final_ds_fixed_score = list(q_to_ds_fixed_score.values())
    DS_fixed = sum(final_ds_fixed_score)/len(final_ds_fixed_score)
    # print(f"{t};{AL:.4f};{DS:.4f}")
    return AL, DS, AL_lenient, DS_fixed, q_to_state_stats

def evaluate(path_to_predictions: Union[str, List[str]], model: str = 'None', dataset: str = 'None', task: str = 'None', analyze_score: bool = False, t: int = 2):
    is_novel = 'novel' in path_to_predictions if isinstance(path_to_predictions, str) else 'novel' in path_to_predictions[0]
    
    qids_to_answer, qids_to_type = filter_out_qas(is_novel=is_novel, path_to_predictions=path_to_predictions)
    data, len_original_file = read_jsonl(path_to_predictions, qids_to_answer, is_novel)
    q_to_chunk_babi, qid_to_type = data_q_to_chunk(data, is_novel, qids_to_answer, qids_to_type)
    
    complex_acc, simple_acc = calculate_acc_score(q_to_chunk_babi)
    AL, DS, AL_lenient, DS_fixed, q_to_state_stats = calculate_simple_score(q_to_chunk_babi, t)

    to_print_key = "Model;Dataset;Task;Original Length;Filtered Length;Acc;old_DS;old_AL;old_AL_lenient;old_DS_fixed;"
    to_print = f"{model};{dataset};{task};{len_original_file};{len(data)};"
    to_print += f"{simple_acc:.4f};{DS:.4f};{AL:.4f};{AL_lenient:.4f};{DS_fixed:.4f};"

    if analyze_score:   
        ###### per question type ######
        all_types = sorted(set(qid_to_type.values()))
        for qtype in all_types:
            qtype_data = {qid: data for qid, data in q_to_chunk_babi.items() if qid_to_type[qid] == qtype}
            _, qtype_acc = calculate_acc_score(qtype_data)
            qtype_al, qtype_ds, qtype_al_lenient, qtype_ds_fixed, q_to_state_stats_ = calculate_simple_score(qtype_data)
            to_print_key += f"q_{qtype};"
            to_print += f"{qtype_acc:.4f};"
            to_print_key += f"q_old_{qtype}_al;q_old_{qtype}_ds_fixed;"
            to_print += f"{qtype_al:.4f};{qtype_ds_fixed:.4f};"
            

        ###### per number of state transitions ######
        ###### per state duration ######
        percentile_values = [3, 5, 20] if not is_novel else [3, 4, 19]
        
        result_num_transitions, _ = analyze_per_state_transition(q_to_state_stats, percentile_values)
        sorted_keys = sorted(result_num_transitions.keys())
        for k in sorted_keys:
            v = result_num_transitions[k]
            to_print_key += f"num_trans_{k};"
            to_print += f"{v['acc']:.4f};"

    print(to_print)

    return simple_acc, DS, AL, DS_fixed

def analyze_per_state_transition(q_to_state_stats, given_percentile_values: List[int] = None):
    num_transitions = []
    state_durations = [] 

    
    for qid, state_stats in q_to_state_stats.items():
        
        num_state_transitions = sum([len(stats['durations']) for stats in state_stats.values()])
        num_transitions.append(num_state_transitions)
        for gt, stats in state_stats.items():
            for duration in stats['durations']:
                state_durations.append(duration)

    num_transitions_percentiles = np.percentile(num_transitions, [25, 50, 75, 100]) if given_percentile_values is None else given_percentile_values
    state_durations_percentiles = np.percentile(state_durations, [25, 50, 75, 100])
    
    # import pdb; pdb.set_trace()
    
    num_transitions_to_scores = defaultdict(lambda: {
        'acc': [],
        'latency': [],
        'distraction': [],
        'latency_lenient': []
    })
    state_durations_to_scores = defaultdict(lambda: {
        'acc': [],
        'latency': [],
        'distraction': [],
        'latency_lenient': []
    })
    for qid, state_stats in q_to_state_stats.items():        
        num_state_transitions = sum([len(stats['durations']) for stats in state_stats.values()])
        if num_state_transitions <= num_transitions_percentiles[0]: key = f'1/{len(num_transitions_percentiles)}'
        elif num_state_transitions <= num_transitions_percentiles[1]: key = f'2/{len(num_transitions_percentiles)}'
        elif num_state_transitions <= num_transitions_percentiles[2]: key = f'3/{len(num_transitions_percentiles)}'
        else: key = '75-100'


        all_correct_num = sum([stat for stats in state_stats.values() for stat in stats['correct_num']])
        all_duration = sum([stat for stats in state_stats.values() for stat in stats['durations']])
        all_latency = sum([stat for stats in state_stats.values() for stat in stats['latencies']])
        all_distraction = sum([stat for stats in state_stats.values() for stat in stats['distractions']])

        al = all_latency / all_duration
        ds = all_distraction/all_duration
        ac = all_correct_num/all_duration

        num_transitions_to_scores[key]['acc'].append(ac)
        num_transitions_to_scores[key]['latency'].append(al)
        num_transitions_to_scores[key]['distraction'].append(ds)

    result_num_transitions = {}
    sorted_keys = sorted(num_transitions_to_scores.keys())
    for k in sorted_keys:
        v = num_transitions_to_scores[k]
        result_num_transitions[k] = {
            "acc": np.mean(v['acc']),
            "latency": np.mean(v['latency']),
            "distraction": np.mean(v['distraction'])
        }
    
    result_state_durations = {}
    return result_num_transitions, result_state_durations


       
        
def filter_out_qas(is_novel: bool, path_to_predictions: Union[str, List[str]]):
    
    def to_option(ind):
        char = chr(ord('A') + ind)[0]
        return char
           
    qids_to_answer = {}
    qids_to_type = {}
    random_perf = []
    if is_novel:
        file_list = [f"data_processed/ours/qas_final_public/{file}" for file in os.listdir("data_processed/ours/qas_final_public")]
        # file_list = [f"data_processed/ours/qas_final/{file}" for file in os.listdir("data_processed/ours/qas_final")]
        # file_list.extend([f"data_processed/ours/qas_final_public/{file}" for file in os.listdir("data_processed/ours/qas_final_public")])
        file_list = sorted(file_list)
        # assert len(file_list) == 39
        for file in file_list:
            book_id = file.split("/")[-1].split(".")[0]
            with open(file, "r") as f:
                data = json.load(f)
            qids = {}
            chunk_texts = data['qa_dict']['chunks']
            for chunk_idx in range(len(chunk_texts)):
                for question_idx, (question, answers) in enumerate(data['qa_dict']['qas'].items()):
                    qid = answers['question_id'].split("_")[-1]

                    gt_answer = answers['chunk_to_answer'][str(chunk_idx)] 
                    gt_answer_options = to_option(answers['options'].index(gt_answer))
                    qid_ = question.strip() if answers['question_id'].startswith("C12") else qid
                    query_id = f"{book_id}_{chunk_idx}_{qid_}"
                    # query_id = f"{book_id}_{chunk_idx}_{qid}"
                    qids_to_answer[query_id] = [gt_answer_options]

                    book_type = 'short' if len(chunk_texts) <= 64 else ('mid' if len(chunk_texts) <= 128 else 'long')
                    # book_type = 'copyright' if answers['question_id'].startswith("C") else 'public'
                    qids_to_type[query_id] = book_type

                    # random_perf.append(1/len(answers['options']))
                    
            #         qids[f"{book_id}_{qid}"] = [gt_answer]
            # print("-"*50)
            # print("Number of qids: ", len(qids))
            # print(qids.keys(), "\n\n\n\n")
        # print(f"Random Performance: {sum(random_perf)/len(random_perf)} out of {len(random_perf)}")
        # assert len(qids_to_answer) == 67497
        print("Number of qids: ", len(qids_to_answer))
        return qids_to_answer, qids_to_type
    else:
        if isinstance(path_to_predictions, str):
            path_to_predictions = [path_to_predictions]
        is_expanded = any('filter20' in path for path in path_to_predictions) or any('Agentic' in path for path in path_to_predictions)
        if is_expanded:
            # print("Expanded mode")
            path_to_corpus = 'data_processed/babilong_final/babilong-ck/babilong-ck.booklen_128k_chunklen_2000.filter20.extended_answer.json'
            
            with open(path_to_corpus, 'r') as f:
                final_babi = json.load(f)   

            qids_to_type = {}
            for fact_idx, fact_set in enumerate(final_babi['chunk_and_qa']):
                for chunk_idx in fact_set['qa_dict']['chunks'].keys(): # starts from 0
                    for question_idx, (question_text, gt_answers) in enumerate(fact_set['qa_dict']['qas'].items()):
                        gt_answer = gt_answers[str(chunk_idx)]
                        query_id= f"{fact_idx}_{chunk_idx}_{question_text}"
                        qids_to_answer[query_id] = gt_answer 
                        qids_to_type[query_id] = fact_set['qa_dict']['qas_type'][question_text]
            
        else:
            path_to_corpus = 'data_processed/babilong_final/babilong-ck/babilong-ck.booklen_128k_chunklen_2000.extended_answer.json'
            path_to_type = 'data_processed/babilong_final/filtered.total_qa_set.subset.json'

            with open(path_to_corpus, 'r') as f:
                final_babi = json.load(f)   
                
            subset_qas = {}
            with open(path_to_type, 'r', encoding='utf-8') as f:
                data_subset = json.load(f)
            for fact_idx, factset in enumerate(data_subset):
                for typ, qas in factset['replaced_subset_qas'].items():
                    for qa in qas:
                        subset_qas[f"{fact_idx}_{qa['question']}"] = typ  
                        
            qids_to_type = {}
            for fact_idx, fact_set in enumerate(final_babi['chunk_and_qa']):
                for chunk_idx in fact_set['qa_dict']['chunks'].keys(): # starts from 0
                    for question_idx, (question_text, gt_answers) in enumerate(fact_set['qa_dict']['qas'].items()):
                        if f"{fact_idx}_{question_text}" in subset_qas:
                            gt_answer = gt_answers[str(chunk_idx)]
                            
                            query_id= f"{fact_idx}_{chunk_idx}_{question_text}"
                            qids_to_answer[query_id] = gt_answer 
                            qids_to_type[query_id] = fact_set['qa_dict']['qas_type'][question_text]
                            
        # print("Number of qids: ", len(qids_to_answer))
        return qids_to_answer, qids_to_type


if __name__ == "__main__":
    paths = [
        "output/Qwen/Qwen3-30B-A3B-Instruct-2507/babilong/babilong-ck.booklen_128k_chunklen_2000-cumulative-common-format-cascade.jsonl",
        "output/Qwen/Qwen3-30B-A3B-Instruct-2507/babilong/babilong-ck.booklen_128k_chunklen_2000-cumulative-common-format-filter20.jsonl"
      ]
    evaluate(paths, analyze_score=True)
    paths = [
        "output/Qwen/Qwen3-30B-A3B-Instruct-2507/novel_strict/novel-copyright-ALL-rolling-128-strictformat.jsonl",
        "output/Qwen/Qwen3-30B-A3B-Instruct-2507/novel_strict/novel-public-ALL-rolling-128-strictformat.jsonl"
      ]
    # evaluate(paths, analyze_score=True)
    paths = [
        "output/Gemini(non-thinking)/2p5-pro/babilong/babilong_gemini_2p5_pro_output_NO_THINKING.jsonl",
        "output/Gemini(non-thinking)/2p5-pro/babilong/babilong_concat_gemini_2p5_pro_output.jsonl"
    ]
    # evaluate(paths, analyze_score=True)
    # evaluate(path)
