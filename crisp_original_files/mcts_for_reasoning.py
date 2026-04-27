# Licensed under the MIT license.
import sys
sys.path.append(".")
import numpy as np, os, random, json, math
from tqdm import trange
from typing import List, Dict, Tuple
from copy import deepcopy
from mcts_backbone import MCTS_Searcher, MCTS_Node
from utils.mcts_utils import (
    Node_Type,
    GeneratorError,
    reach_terminal_ost_step,
    concat_ost_steps,
    make_hint,
    print_tree_from_root
)
from utils.load_data import load_prompt, format_prompt
count = 0

def verbose_print(s: str, verbose: bool):
    # if verbose:
    #     print(s)
    pass

class Generator:
    """Generator generates children nodes"""

    def __init__(self, args, model, reward) -> None:
        self.model = model
        self.evaluator = reward
        self.num_a1_steps = args.num_a1_steps
        self.mcts_num_last_votes = args.mcts_num_last_votes
        self.fewshot_cot_prompt = load_prompt(dataset=args.dataset, method='mcts_cot')
        self.fewshot_ost_prompt  = load_prompt(dataset=args.dataset, method='mcts_ost')

    def _get_most_likely_answer(self, user_question, io_output_list: List[str]) -> Tuple[str, float]:
        assert len(io_output_list) > 0

        # if len(io_output_list) == 1:
        #     most_confident_answer_full_completion = io_output_list[0]
        #     confidence = 1
        # else:
        _, most_confident_answer_full_completion, confidence = self.evaluator.find_most_confident_answer(
            user_question,
            io_output_list
        )
        # assert confidence > 0

        return most_confident_answer_full_completion, confidence

    def _fewshot_cot_answer_question(self, question: str, num_return: int, hint: str = None):        
        fewshot_cot_prompt = self.fewshot_cot_prompt
        item = {'question': question}
        if hint:
            io_input = format_prompt(fewshot_cot_prompt, item) + hint
        else:
            io_input = format_prompt(fewshot_cot_prompt, item)
        sessions = io_input.split('####')
        if self.model.is_chat:
            inputs = [{"role": "system", "content": sessions[0]}]
        else:
            inputs = sessions[0]
        for session in sessions[1:]:
            if len(session.split('# Reasoning:')) <= 1:
                if self.model.is_chat:
                    inputs += [{"role": "user", "content": session}, {"role": "assistant", "content": ""}]
                else:
                    inputs += session
            else:
                if len(session.split('# Reasoning:')) > 2:
                    content = session.split('# Reasoning:')
                    user_content = content[0]
                    assistant_content = ('').join(content[1:])
                else:
                    # print(session)
                    user_content, assistant_content = session.split('# Reasoning:')
                assistant_content = '# Reasoning:' + assistant_content
                if self.model.is_chat:
                    inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
                else:
                    inputs += user_content + assistant_content
          # inputs = inputs[:-1]
        
        output_list = self.model.generate(
            input = inputs,
            max_tokens=1024,
            sample_cnt=num_return,
            continue_generate=True
        )
        global count 
        for response in output_list:
            count += len(response)
        if hint:
            output_list = [hint + output.strip() for output in output_list]  #! cleaning
        else:
            output_list = [output.strip() for output in output_list]  
        return io_input, output_list

    def generate_direct_answers(self, user_question: str, hint: str):
        direct_answer_list, value_list = [], []

        #! few shot cot
        num_return = self.mcts_num_last_votes
        io_input, cleaned_io_output_list = self._fewshot_cot_answer_question(
            question=user_question, num_return=num_return, hint=hint
        )
    
        try:
            most_likely_answer, likelihood = self._get_most_likely_answer(user_question, cleaned_io_output_list)
        except Exception as e:
            raise GeneratorError(
                source="generate direct answer from: few shot cot",
                io_input=io_input,
                io_output_list=cleaned_io_output_list,
            )

        direct_answer_list.append(most_likely_answer)
        value_list.append(likelihood)

        return direct_answer_list, value_list


    def generate_ost_step(
        self,
        user_question: str,
        solution_trace: Dict[int, Dict[str, str]],
    ):
        ost_step_list = []
        existing_ost_steps, next_ost_step_id = concat_ost_steps(solution_trace)
        fewshot_ost_prompt = self.fewshot_ost_prompt
        # print(fewshot_ost_prompt)
        item = {'question':user_question}
        io_input = (
            format_prompt(fewshot_ost_prompt, item) 
            + existing_ost_steps
            + f"Step {next_ost_step_id}:"
        )
        # print(io_input)
        sessions = io_input.split('####')
        if self.model.is_chat:
            inputs = [{"role": "system", "content": sessions[0]}]
        else:
            inputs = sessions[0]
        for session in sessions[1:]:
            if len(session.split('# Reasoning:')) <= 1:
                if self.model.is_chat:
                    inputs += [{"role": "user", "content": session}, {"role": "assistant", "content": ""}]
                else:
                    inputs += session
            else:
                if len(session.split('# Reasoning:')) > 2:
                    content = session.split('# Reasoning:')
                    user_content = content[0]
                    assistant_content = ('').join(content[1:])
                else:
                    # print(session)
                    user_content, assistant_content = session.split('# Reasoning:')
                assistant_content = '# Reasoning:' + assistant_content
                if self.model.is_chat:
                    inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
                else:
                    inputs += user_content + assistant_content
        # print(existing_ost_steps)
        # print('>>>>>>>')
        io_output_list = self.model.generate(
            input=inputs, 
            max_tokens=256, 
            sample_cnt=self.num_a1_steps, 
            continue_generate=True,
            stop_tokens=['.\n']
        )
        global count
        for res in io_output_list:
            count += len(res)
        ost_step_list = [io_output.split('\nStep')[0].strip() for io_output in io_output_list]
    
        # print(ost_step_list)
        # value_list = []
        # direct_answer_list = []
        # for ost_step in ost_step_list:
        #     if reach_terminal_ost_step(ost_step):
        #         io_output_list = [existing_ost_steps + f"Step {next_ost_step_id}: {ost_step}"]
        #         most_likely_answer, likelihood = self._get_most_likely_answer(user_question=user_question, io_output_list=io_output_list)
        #     else:
        #         most_likely_answer = None 
        #         likelihood = None 
        #     value_list.append(likelihood)
        #     direct_answer_list.append(most_likely_answer)
        return ost_step_list


class Reasoning_MCTS_Node(MCTS_Node):
    def __init__(
        self,
        parent: "Reasoning_MCTS_Node",
        depth: int,
        node_type: Node_Type,
        verbose: bool = False,
        # --- For instantiating root node ---
        node_value: float = None,
        generator: Generator = None,
        user_question: str = None,
        max_depth_allowed: int = None,
        # --- For instantiating DIRECT_ANSWER node ---
        direct_answer: str = None,
        # --------------------------------------------
        # --- For instantiating OST_STEP node ---
        ost_step: str = None,
        # ---------------------------------------
    ) -> None:
        """params:
        subquestion: the node is proposing a new subquestion
        subanswer: the answer corresponding to the new subquestion the node proposed
        re_subanswer: the node is proposing a new subanswer to the parent's subquestion
        """
        super().__init__()

        #! attributes
        self.parent = parent  # if parent is None, then the node is the root
        self.children: List["Reasoning_MCTS_Node"] = []
        self.depth = depth
        self.node_type = node_type
        self.node_value = node_value
        self.direct_answer = direct_answer
        self.ost_step = ost_step

        if parent is None:  # root
            self.verbose = False
            self.user_question = user_question
            self.generator = generator
            self.max_depth_allowed = max_depth_allowed
        else:  # inherit from parent
            self.verbose = False
            self.user_question = parent.user_question
            self.generator = parent.generator
            self.max_depth_allowed = parent.max_depth_allowed


        #! record number of one-step thought steps till now
        if parent is None:  # root
            self.ost_step_counter = 0
        else:
            if node_type is Node_Type.OST_STEP:
                self.ost_step_counter = parent.ost_step_counter + 1
            else:
                self.ost_step_counter = parent.ost_step_counter

        #! record solution trace from root to the current node. key: subquestion id
        if parent is None:  # root
            assert self.node_type is Node_Type.USER_QUESTION
            self.solution_trace: Dict[str, str] = {"user_question": user_question, "ost_step": {}}
        else:
            assert self.node_type is not Node_Type.USER_QUESTION
            self.solution_trace = deepcopy(parent.solution_trace)
            self.solution_trace["ost_step"][self.ost_step_counter] = ost_step

    def __str__(self) -> str:
        type2str = {
            Node_Type.USER_QUESTION: "U",
            Node_Type.REPHRASED_USER_QUESTION: "RU",
            Node_Type.DIRECT_ANSWER: "DA",
            Node_Type.SUBQUESTION: "SQ",
            Node_Type.RE_SUBANSWER: "RS",
            Node_Type.OST_STEP: "TS",
        }
        return f"{type2str[self.node_type]}-{self.id}"

    def _create_children(self):
        def do_action_generate_direct_answers():
            verbose_print(f"---- Generating direct answers for node {self.id}...", self.verbose)

            #! ACTION: generate direct answer for the user question (w/ or w/o hint)
            if self.node_type is not Node_Type.USER_QUESTION:
                hint = make_hint(self.solution_trace, self.node_type)
            else:
                hint = None

            (direct_answer_list, value_list) = self.generator.generate_direct_answers(
                user_question=self.user_question, hint=hint
            )
          
            for direct_answer, value in zip(direct_answer_list, value_list):
                # if np.isnan(value) or value <= 0:
                #     breakpoint()
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.DIRECT_ANSWER,
                        node_value=value,
                        direct_answer=direct_answer,
                    )
                )

        def do_action_generate_ost_step():
            verbose_print(f"---- Generating one-step thought steps for node {self.id}...", self.verbose)

            #! ACTION: generate one-step thought step
            ost_step_list = self.generator.generate_ost_step(
                user_question=self.user_question,
                solution_trace=self.solution_trace,
            )
            for ost_step in ost_step_list:
                self.children.append(
                    Reasoning_MCTS_Node(
                        parent=self,
                        depth=self.depth + 1,
                        node_type=Node_Type.OST_STEP,
                        ost_step=ost_step
                    )
                )
                # print(ost_step)
        if self.depth >= self.max_depth_allowed - 1 or reach_terminal_ost_step(self.ost_step):
            do_action_generate_direct_answers()
        else:
            do_action_generate_ost_step()

        assert self.children
        return self.children

    def is_valid_leaf_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type
        # return (self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step)
        return self.node_type is Node_Type.DIRECT_ANSWER

    def is_valid_solution_node(self):
        #! a valid solution can only be in SUBQUESTION type or DIRECT_ANSWER type or OST_STEP type
        return (
            self.node_type is Node_Type.OST_STEP and reach_terminal_ost_step(self.ost_step)
            or self.node_type is Node_Type.DIRECT_ANSWER
        )

    def find_children(self, rollout_id: int):
        self.children = self.children or self._create_children()
        for child in self.children:
            child.set_rollout_id(rollout_id)
        assert self.children
        return self.children

    def is_terminal(self):
        return self.depth >= self.max_depth_allowed or self.is_valid_leaf_node()

    def calculate_reward(self):
        if self.is_valid_leaf_node():
            assert self.node_value is not None, breakpoint()
            return self.node_value
        else:
            return 0

    def skip_backprop(self):
        return self.node_type is Node_Type.USER_QUESTION


def search_for_answers(args, user_question: str, question_id: int, generator: Generator):
    verbose_print(
        f"********************* Searching for answers to question {question_id} ********************* ", True
    )
    #! build an MCTS searcher
    mcts_searcher = MCTS_Searcher(
        exploration_weight=args.mcts_exploration_weight,
        weight_scheduler=args.mcts_weight_scheduler,
        num_rollouts=args.roll_num,
        discount=args.mcts_discount_factor,
        verbose=True,
    )

    #! build the MCTS tree
    root_node = Reasoning_MCTS_Node(
        parent=None,
        depth=0,
        node_type=Node_Type.USER_QUESTION,
        verbose=True,
        generator=generator,
        user_question=user_question,
        max_depth_allowed=args.max_depth_allowed,
    )

    model_solutions = []
    model_solution_q_values = []
    model_solution_rewards = []
    traces = {}
    for i in (pbar := trange(args.roll_num, disable=True, position=0)):
        roll_out_path, rollout_node = mcts_searcher.do_rollout(root_node, i)
        # _, best_solution, _, chosen_node, all_solution_nodes, all_solutions = stochastic_find_best_solution(
        #     root_node, generator.evaluator
        # )
        model_solutions.append(rollout_node.direct_answer)
        model_solution_rewards.append(rollout_node.node_value)
        q_value = 0.0
        for node in roll_out_path:
            q_value += mcts_searcher.Q[node]
        model_solution_q_values.append(q_value)
        # model_all_solutions += all_solutions
        # print(best_solution)
        # if args.save_tree:
        #     with open(
        #         os.path.join(
        #             args.answer_sheets_dir,
        #             f"Question {question_id:04d} - Rollout {i}.tree",
        #         ),
        #         "w",
        #     ) as f:
        traces[i] = {}
        # if args.roll_num <= 32:
        #     print_tree_from_root(
        #         mcts_searcher=mcts_searcher,
        #         rollout_id=i,
        #         root_node=root_node,
        #         traces = traces
        #     )
        
    bestN_node = root_node
    bestQ_node = root_node
    while not bestN_node.is_terminal():
        if not bestN_node.children:
            break
        bestN_node = max(bestN_node.children, key=lambda x:mcts_searcher.N[x])
    while not bestQ_node.is_terminal():
        if not bestQ_node.children:
            break
        bestQ_node = max(bestQ_node.children, key=lambda x:mcts_searcher.Q[x])
    traces[-1] = {'bestN':bestN_node.direct_answer, 'bestQ':bestQ_node.direct_answer}
    model_solutions = [{'content':res, 'q_value':q_value, 'reward':reward}  for res, q_value, reward in zip(model_solutions, model_solution_q_values, model_solution_rewards)]
    #! record final traces
    # js = [{"trace": node.solution_trace, "rollout_id": node.rollout_id} for node in all_solution_nodes]
    # with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Final Solutions.json"), "w") as f:
    #     json.dump(js, f)

    # js2 = [{"trace": node.solution_trace, "rollout_id": i} for i, node in enumerate(model_rollout_nodes)]
    # with open(os.path.join(args.answer_sheets_dir, f"Question {question_id:04d} - Rollout Solutions.json"), "w") as f:
    #     json.dump(js2, f)
    global count
    traces[-1]['count'] = count
    return model_solutions, traces
