from typing import List, Dict, Tuple
from copy import deepcopy
import numpy as np
from utils.load_data import format_prompt, extract_answer, load_prompt
from utils.mcts_utils import reach_terminal_ost_step
import torch 


class BeamNode(object):
    def __init__(
        self,
        parent: "BeamNode",
        user_question: str = None,
        ost_step: str = None,
        answer: str = None,
        # ---------------------------------------
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! attributes
        self.answer = None 
        self.children: List["BeamNode"] = []
        self.answer = answer
        self.ost_step = ost_step
        if parent is None:  # root
            self.user_question = user_question
            self.solution_trace: List[str] = []
        else:  # inherit from parent
            self.user_question = parent.user_question
            self.solution_trace = deepcopy(parent.solution_trace)
            self.solution_trace.append(ost_step)
            
            
def generate_ost_step(node, model, sample_cnt, remain, args):
    ost_step_list = []
    if remain:
        existing_ost_steps = ('\n').join(node.solution_trace)
        fewshot_ost_prompt = load_prompt(args.dataset, 'mcts_cot')
    elif node.solution_trace:
        solution_trace = [f'Step {i}: ' + node.solution_trace[i] for i in range(len(node.solution_trace))]
        existing_ost_steps = ('\n').join(solution_trace) + f"Step {len(node.solution_trace)+1}: "
        fewshot_ost_prompt = load_prompt(args.dataset, 'mcts_ost')
    else:
        existing_ost_steps = 'Step 1: '
        fewshot_ost_prompt = load_prompt(args.dataset, 'mcts_ost')
    # print(fewshot_ost_prompt)
    item = {'question':node.user_question}
    io_input = format_prompt(fewshot_ost_prompt, item) + existing_ost_steps
    sessions = io_input.split('####')
    inputs = [{"role": "system", "content": sessions[0]}]
    for session in sessions[1:]:
        if len(session.split('# Reasoning:')) <= 1:
            inputs += [{"role": "user", "content": session}, {"role": "assistant", "content": ""}]
        else:
            if len(session.split('# Reasoning:')) > 2:
                content = session.split('# Reasoning:')
                user_content = content[0]
                assistant_content = ('').join(content[1:])
            else:
                user_content, assistant_content = session.split('# Reasoning:')
            assistant_content = '# Reasoning:' + assistant_content
            inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        
    if remain:
        io_output_list = model.generate(
            input=inputs, sample_cnt=sample_cnt, continue_generate=True, temperature=args.temperature
        )
        ost_step_list = [io_output.strip() for io_output in io_output_list]
    else:
        io_output_list = model.generate(
            input=inputs, max_tokens=256, sample_cnt=sample_cnt, stop_tokens=['.\n', '\nStep'], continue_generate=True, temperature=args.temperature
        )
        ost_step_list = [io_output.split('\nStep')[0].strip() for io_output in io_output_list]

    answer_list = []
    for ost_step in ost_step_list:
        if reach_terminal_ost_step(ost_step):
            answer = extract_answer(ost_step, args.dataset)
        else:
            answer = None 
        answer_list.append(answer) 
    return ost_step_list, answer_list


# def find_valid_solution_nodes(root_node):
#     valid_solution_nodes = []

#     def recursion(node):
#         if reach_terminal_ost_step(node.ost_step):
#             valid_solution_nodes.append(node)
#             return

#         if not node.children:  #! no children
#             return

#         for child in node.children:
#             recursion(child)
#     recursion(root_node)  
#     return valid_solution_nodes

def select_top_node(node_list, reward, topk):
    question = node_list[0].user_question
    responses = []
    for node in node_list:
        solution_trace = ('\n').join(node.solution_trace)
        responses.append(solution_trace)

    scores = torch.tensor(reward.score(question, responses, step_reward=False))
    if len(scores) < topk:
        topk = len(scores)
    top_k_values, top_k_indices = torch.topk(scores, topk)
    top_node = [node_list[idx] for idx in top_k_indices]
    return top_node, top_k_values.tolist(), responses, scores.tolist()


def beam_search(args, model, reward, question):
    

    root_node = BeamNode(
        parent=None,
        user_question=question
    )
    depth = 0
    node_list = [root_node]
    traces = {}
    while depth < args.max_depth_allowed:
        new_node_list = []
        for node in node_list:
            if node.answer:
                ost_step_list = [""]
                answer_list = [node.answer]
            else:
                if depth == 0:
                    ost_step_list, answer_list = generate_ost_step(node, model, sample_cnt=args.roll_num, remain=False, args=args)
                elif depth >= args.max_depth_allowed - 1:
                    ost_step_list, answer_list = generate_ost_step(node, model, sample_cnt=args.beam_width, remain=True, args=args)
                else:
                    ost_step_list, answer_list = generate_ost_step(node, model, sample_cnt=args.beam_width, remain=False, args=args)
            for ost_step, direct_answer in zip(ost_step_list, answer_list):
                new_node = BeamNode(
                        parent=node,
                        ost_step=ost_step,
                        answer=direct_answer
                    )
                node.children.append(new_node)
                new_node_list.append(new_node)
        if not new_node_list:
            break
        node_list, node_values, all_response, all_score = select_top_node(new_node_list, reward, args.roll_num)
        traces[depth] = [{'content':all_response[i], 'score':all_score[i]} for i in range(len(all_response))]
        depth += 1

    responses = [('\n').join(node.solution_trace) for node in node_list]
    responses = [{'content':responses[i], 'score':node_values[i]} for i in range(len(node_list))]
    answers = [node.answer for node in node_list]
    for i in range(len(answers)):
        if not answers[i]:
            answers[i] = extract_answer(responses[i]['content'], args.dataset)

    return responses, answers, traces