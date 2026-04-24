import argparse
import os
import json
import numpy as np
from utils.model import ModelWrapper
from utils.load_data import DataLoader, extract_answer, load_json_data
from transformers import set_seed
from tqdm import tqdm
import random
from utils.reward import reward_factory
from utils.eval import is_equiv
from mcts_for_reasoning import Generator, search_for_answers
from beam_search import beam_search
from crisp_reason import crisp_reason
import time

def beam_reason(args, item, model, reward):
    responses, answer, traces = beam_search(args=args, model=model, reward=reward, question=item['question'])
    # pred = max(answer, key=answer.count)
    pred = answer[0]
    return responses, answer, pred, traces


def mcts_reason(args, item, model, reward, select='sc'):
    generator = Generator(args, model, reward)
    problem_id, problem = item["id"], item["question"]
    # print(problem)
    model_solutions, traces = search_for_answers(
        args=args, user_question=problem, question_id= problem_id, generator=generator
    )
    # print(model_all_solutions)
    answer = [extract_answer(item['content'], dataset=args.dataset) for item in model_solutions]
    if select == 'sc':
        pred = max(answer, key=answer.count)
    elif select == 'bestN':
        pred = extract_answer(traces[-1]['bestN'])
    elif select == 'bestQ':
        pred = extract_answer(traces[-1]['bestQ'])
    elif select == 'q_value':
        solution = max(model_solutions, key=lambda x: x['q_value'])['content']
        pred = extract_answer(solution, dataset=args.dataset)
    else:
        solution = max(model_solutions, key=lambda x: x['reward'])['content']
        pred = extract_answer(solution, dataset=args.dataset)
    
    # pred = answer[-1]
    
    return model_solutions, answer, pred, traces

def reward_sc_reason(dataset, model, inputs, reward_model, nums=10, scores=None):
    if isinstance(inputs, dict):
        question = inputs['question']
        response = inputs['response']
        answer = inputs['answer']
    else:
        question = inputs[-1]['content']
        response = model.generate(inputs, sample_cnt=nums)
        answer = [extract_answer(output=res, dataset=dataset) for res in response]
    if not scores:
        scores = reward_model.score(question, response, step_reward=False)
    # if reward_model.step_reward:
    #     scores = [np.prod(np.array(score)) for score in scores]
    coef = {}
    for i in range(nums):
        if answer[i] not in coef.keys():
            coef[answer[i]] = scores[i]
        else:
            coef[answer[i]] += scores[i]
    pred = max(coef, key=lambda x: coef[x])
    response = [{'content': response[i], 'score':float(scores[i]), 'coef':float(coef[answer[i]])} for i in range(nums)]

    return response, answer, pred, None  

def sc_reason(dataset, model, inputs, nums, temperature):
    if dataset.startswith('gpqa') or model.is_ds:
        response = model.generate(inputs, sample_cnt=nums, max_tokens=5000, temperature=temperature)
    else:
        response = model.generate(inputs, sample_cnt=nums, temperature=temperature)

    answer = [extract_answer(output=res, dataset=dataset) for res in response]
    pred = max(answer, key=answer.count)

    return response, answer, pred, trace

def bestn_reason(dataset, model, inputs, reward_model, nums=10):
    if isinstance(inputs, dict):
        question = inputs['question']
        if isinstance(inputs['response'][0], dict):
            response = [tup['content'] for tup in inputs['response']]
        else:
            response = inputs['response'][:nums]
    else:
        question = inputs[-1]['content']
        response = model.generate(inputs, sample_cnt=nums)

    scores = reward_model.score(question, response, step_reward=False, agg=agg)
    answer = [extract_answer(output=res, dataset=dataset) for res in response]
    pred = answer[np.argmax(np.array(scores))]
    response = [{'content': response[i], 'score':float(scores[i])} for i in range(nums)]

    return response, answer, pred, trace  

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Qwen2_5_3b_chat')
    parser.add_argument('--n_samples', type=int, default=5)
    parser.add_argument('--n_examples', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='gsm8k')
    parser.add_argument('--method', type=str, default='cot')
    parser.add_argument('--roll_num', type=int, default=16)
    parser.add_argument('--reward', type=str, default=None)
    parser.add_argument('--remote', action='store_true')
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--agg', choices=["last", "avg", None], default='last')
    
    parser.add_argument("--num_votes", type=int, default=10)
    parser.add_argument("--max_depth_allowed", type=int, default=3)
    parser.add_argument("--mcts_discount_factor", type=float, default=1.0)
    parser.add_argument("--mcts_exploration_weight", type=float, default=1.0)
    parser.add_argument("--mcts_weight_scheduler", choices=["exp", "lin", "const"], default="const")
    parser.add_argument("--mcts_num_last_votes", type=int, default=1)
    parser.add_argument("--num_a1_steps", type=int, default=5)
    parser.add_argument("--select", type=str, default='reward')
    
    parser.add_argument("--beam_width", type=int, default=2)
    
    parser.add_argument("--sample_algo", choices=["direct", "mcts"], default="direct")
    parser.add_argument("--sample_reward", type=str, default=None)
    parser.add_argument("--ablation", type=str, default=None)
    
    parser.add_argument("--max_cls", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    
    model_name = args.model
    n_samples = args.n_samples
    n_examples = args.n_examples
    dataset = args.dataset 
    method = args.method
    roll_num = args.roll_num
    reward = args.reward
    remote = args.remote
    split = args.split
    test = args.test
    temperature = args.temperature
    select = args.select
    agg = args.agg
    sample_algo = args.sample_algo
    sample_reward = args.sample_reward
    ablation = args.ablation
    
    if sample_reward:
        args.mcts_exploration_weight = args.mcts_exploration_weight * 50 if sample_reward == 'skywork' else args.mcts_exploration_weight
    else:
        args.mcts_exploration_weight = args.mcts_exploration_weight * 50 if reward == 'skywork' else args.mcts_exploration_weight

    dataloader = DataLoader(dataset=dataset, n_samples=n_samples)
    data = dataloader.load_data(method='cot', n_examples=n_examples, mode=split)
    
    model_init = True    
    reward_init = True     
    if reward:
        if reward_init:
            reward_model = reward_factory(reward, remote, dataset)
        else:
            reward_model = None 

   
    model = ModelWrapper(model_name, remote, model_init)
    result = []
    correct = 0
    if method in ['cot', 'sc', 'bestn', 'reward_sc', 'mcts', 'beam', 'crisp']:
        split_str = '# Reasoning:'
    else:
        split_str = '# Answer:'

    if test:
        data = data[:1]

    cnt = 0
    response = []
    start_time = time.time()
    for item in tqdm(data):
        if method in ['cot', 'sc', 'reward_sc', 'bestn']:
            sessions = item['question'].split('####')
            if model.is_mistral or model.is_gemma or model.is_o1:
                inputs = []
            elif model.is_qwen and not model.is_chat:
                inputs = sessions[0]
            else:
                inputs = [{"role": "system", "content": sessions[0]}]
            for session in sessions[1:]:
                user_content, assistant_content = session.split(split_str)
                assistant_content = split_str + assistant_content
                if model.is_chat:
                    inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
                else:
                    inputs += user_content + assistant_content
            if model.is_chat: 
                inputs = inputs[:-1]
            if model.is_mistral or model.is_gemma or model.is_o1:
                inputs[0]['content'] = sessions[0] + '\n' + inputs[0]['content']
        else:
            item['question'] = item['raw_question']
        # print(inputs)

        
        if method == 'sc':
            response, answer, pred, traces = sc_reason(dataset, model, inputs, roll_num, temperature) 
        elif method == 'crisp':
            response, answer, pred, traces = crisp_reason(model, reward_model, item['question'], args) 
        elif method == 'bestn':
            # if sample_results:
            #     inputs = sample_results[item['id']]
            response, answer, pred, traces = bestn_reason(dataset, model, inputs, reward_model, roll_num)
        elif method == 'reward_sc':
            response, answer, pred, traces = reward_sc_reason(dataset, model, inputs, reward_model, roll_num, scores)
        elif method == 'mcts':
            response, answer, pred, traces = mcts_reason(args, item, model, reward_model, select)   
        elif method == 'beam':
            response, answer, pred, traces = beam_reason(args, item, model, reward_model) 
        else:
            response = model.generate(inputs)
            answer = extract_answer(output=response, dataset=dataset)
            pred = answer
            traces = None 
        if isinstance(answer, list):
            corrects = [is_equiv(ans, item['answer'], dataset) for ans in answer]
        else:
            corrects = is_equiv(answer, item['answer'], dataset)
        
        cor_flag = is_equiv(pred, item['answer'], dataset)
        if 'reason' not in item.keys():
            item['reason'] = None 
        msg = {'id':item['id'], 'question':item['raw_question'], 'reason':item['reason'], 'response':response, 'answer':answer, 'pred':pred, 'label':item['answer'], 'corrects':corrects, 'cor_flag':cor_flag, 'trace':traces}
        result.append(msg)
        correct += int(cor_flag)
        cnt += 1
    
    avg_time = (time.time() - start_time) / cnt
    result.append({'acc': correct / cnt, 'time':avg_time})
    result_dir = f'./result/{dataset}/{model_name}/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if method == 'sc':
        result_path = os.path.join(result_dir, f'sc{roll_num}_t{temperature}_e{n_examples}_{n_samples}.json')
    elif method == 'bestn':
        if sample_algo == 'direct':
            result_path = os.path.join(result_dir, f'best{roll_num}_t{temperature}_{reward}_e{n_examples}_{n_samples}.json')
        else:
            result_path = os.path.join(result_dir, f'best{roll_num}_mcts_{sample_reward}_t{temperature}_{reward}_e{n_examples}_{n_samples}.json')
    elif method == 'reward_sc':
        result_path = os.path.join(result_dir, f'sc{roll_num}_t{temperature}_{reward}_e{n_examples}_{n_samples}.json')
    elif method == 'mcts':
        result_path = os.path.join(result_dir, f'mcts{roll_num}_t{temperature}_d{args.max_depth_allowed}_w{args.mcts_exploration_weight}_{args.num_a1_steps}_{args.mcts_num_last_votes}_{reward}_{select}_e{n_examples}_{n_samples}.json')
    elif method == 'beam':
        result_path = os.path.join(result_dir, f'beam{roll_num}_t{temperature}_d{args.max_depth_allowed}_{args.beam_width}_{reward}_e{n_examples}_{n_samples}.json')
    elif method == 'crisp':
        if ablation:
            result_path = os.path.join(result_dir, f'crisp{roll_num}_w{args.beam_width}_d{args.max_depth_allowed}_{reward}_e{n_examples}_{n_samples}_{ablation}.json')
        else:
            result_path = os.path.join(result_dir, f'crisp{roll_num}_w{args.beam_width}_d{args.max_depth_allowed}_{reward}_e{n_examples}_{n_samples}_seed{args.seed}.json')
    else:   
        result_path = os.path.join(result_dir, f'{method}_e{n_examples}_{n_samples}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=4)
        f.close()