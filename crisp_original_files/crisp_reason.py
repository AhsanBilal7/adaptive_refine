from utils.load_data import format_prompt, extract_answer, load_prompt
from utils.mcts_utils import reach_terminal_ost_step
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

node_cnt = 0
class Generator:
    """Generator generates children nodes"""

    def __init__(self, model, reward, args) -> None:
        self.model = model
        self.reward = reward
        self.direct_nums = args.direct_sample_nums
        self.crisp_nums = args.crisp_sample_nums
        self.prompt  = load_prompt(dataset=args.dataset, method='mcts_ost')
        self.dataset = args.dataset
        self.temperature = args.temperature

    def generate_response(
        self,
        question: str,
        prefix: list = None
    ):
        prompt = self.prompt
        # print(fewshot_ost_prompt)
        item = {'question':question}
        if prefix:
            prefix_str = ('\nStep').join(prefix).split('# Reasoning:')[-1]
            io_input = (
                format_prompt(prompt, item) 
                + prefix_str
                + f"\nStep {len(prefix)}:"
            )
        else:
            io_input = format_prompt(prompt, item)
        # print(io_input)
        sessions = io_input.split('####')
        if self.model.is_chat:
            inputs = [{"role": "system", "content": sessions[0]}]
            for session in sessions[1:]:
                if len(session.split('# Reasoning:')) <= 1:
                    inputs += [{"role": "user", "content": session}, {"role": "assistant", "content": ""}]
                else:
                    user_content, assistant_content = session.split('# Reasoning:')
                    assistant_content = '# Reasoning:' + assistant_content
                    inputs += [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
        else:
            inputs = sessions[0]    
            for session in sessions[1:]:
                if len(session.split('# Reasoning:')) <= 1:
                    inputs += session
                else:
                    user_content, assistant_content = session.split('# Reasoning:')
                    assistant_content = '# Reasoning:' + assistant_content
                    inputs += user_content + assistant_content

        
        if self.model.is_ds:
            max_tokens = 5000
        else:
            max_tokens = 2048
        if prefix: 
            output_list = self.model.generate(
                input=inputs, 
                max_tokens=max_tokens, 
                sample_cnt=self.crisp_nums,
                continue_generate=True,
                temperature=self.temperature
            )
            output_list = [('\nStep').join(prefix) + f"\nStep {len(prefix)}:" + output.strip() for output in output_list]
        else:
            output_list = self.model.generate(
                input=inputs, 
                max_tokens=max_tokens, 
                sample_cnt=self.direct_nums,
                temperature=self.temperature
            )
        responses = [output.strip() for output in output_list]

        scores = self.reward.score(question=question, responses=responses, agg='last', step_reward=False)
        answers = [extract_answer(content, self.dataset) for content in responses]
        return responses, scores, answers


class CRISP_Node(object):
    def __init__(
        self,
        parent: "CRISP_Node",
        depth: int,
        question: str = None,
        content: str = None,
        answer: str = None,
        value: float = None,
        generator: Generator = None,
        
        # ---------------------------------------
    ) -> None:
        #! attributes
        global node_cnt
        self.id = node_cnt
        node_cnt += 1
        self.depth = depth
        self.answer = answer
        self.question = question 
        self.children: List[CRISP_Node] = []
        self.answer = answer
        self.content = content 
        self.value = value
        
        if parent:
            self.question = parent.question 
            self.generator = parent.generator
        else:
            self.question = question
            self.generator = generator
        
    def create_children(self):
        if self.depth == 0:
            responses, values, answers = self.generator.generate_response(
                question=self.question
            )
        else:
            step_ls = [item.strip() for item in self.content.split('\nStep') if item]
            prefix = step_ls[:self.depth+1]
            responses, values, answers = self.generator.generate_response(
                question=self.question, prefix=prefix
            )
            #! ACTION: generate one-step thought step
            
        for content, value, answer in zip(responses, values, answers):
            self.children.append(
                CRISP_Node(
                    parent=self,
                    depth=self.depth + 1,
                    content=content,
                    value=value,
                    answer=answer
                )
            )
    
        return self.children
       
class CRISP_Searcher:

    def __init__(
        self,
        topk: int,
        max_depth: int, 
        ablation: bool,
        max_cls: int
    ):
        self.Q: Dict[CRISP_Node, float] = defaultdict(float)  # total reward of each node
        self.N: Dict[str, int] = defaultdict(int)  # total visit count for each node
        self.parent2children: Dict[CRISP_Node, List[CRISP_Node]] = dict()  # children of each node
        self.scores: Dict[str, float] = defaultdict(float)
        #! explored = expanded + simulated, i.e. has seen terminal at least once, i.e. we can calculate its UCT value, i.e. has Q and N
        self.max_cls = max_cls
        self.topk = topk
        self.max_depth = max_depth
        self.nodes = []
        self.explored_nodes = []
        self.ablation = ablation
        global node_cnt
        node_cnt = 0
        
    def do_rollout(self, root_node: CRISP_Node):
        depth = 0
        node_list = [root_node]
        while depth < self.max_depth:
            for node in node_list:
                self.nodes += node.create_children() 
            node_list = self.select_node()
            if self.is_terminal() and self.ablation != 'exploration':
                return self.get_final_answer()
            depth += 1
        return self.get_final_answer()

    def select_node(self):
        for node in self.nodes:
            if node in self.explored_nodes:
                continue
            self.explored_nodes.append(node)
            if not node.answer:
                continue
            self.Q[node] = node.value
            self.N[node.answer] += 1
            
            
        self.scores = defaultdict(float)
        depth = max([node.depth for node in self.nodes])
        # max_score = np.array(list(self.Q.values())).max()
        # min_score = np.array(list(self.Q.values())).min()
        mean_score = np.mean(np.array(list(self.Q.values())))
        std_score = np.std(np.array(list(self.Q.values())))
        new_node_dic = {}
        for node in self.nodes:
            if not node.answer:
                continue
            if self.ablation == 'expansion' and node.depth != depth:
                continue
            self.scores[node.answer] += (self.Q[node] - mean_score) / std_score
            # self.scores[node.answer] += (self.Q[node])
            if node not in self.scores.keys():
                new_node_dic[node.answer] = node
            else:
                if node.value > new_node_dic[node.answer].value:
                    new_node_dic[node.answer] = node    
        if self.ablation == 'selection': 
            good_answers = sorted(self.Q.items(), key=lambda x:x[1], reverse=True)[:self.topk]   
            new_node_ls = [item[0] for item in good_answers]
            good_answers = [item[0].answer for item in good_answers]
        else:    
            good_answers = sorted(self.scores.items(), key=lambda x:x[1], reverse=True)[:self.topk]
            good_answers = [item[0] for item in good_answers]
            new_node_ls = [item[1] for item in new_node_dic.items() if item[0] in good_answers]
        
        return new_node_ls

    def is_terminal(self):
        if len(self.N.keys()) <= self.max_cls:
            return True 

        return False 


    def get_final_answer(self):
        if not self.scores:
            return None 
        if self.ablation == 'selection':
            return sorted(self.Q.items(), key=lambda x:x[1], reverse=True)[0][0].answer
        else:
            return sorted(self.scores.items(), key=lambda x:x[1], reverse=True)[0][0]
    

def print_tree_from_root(searcher):
    traces = {}
    for node in searcher.nodes:
        info = {
                'depth':node.depth, 
                'content':node.content,
                'answer':node.answer,
                }

        children = [item.id for item in node.children] if node.children else None 
        score = searcher.scores[node.answer] if node.answer else None 
        q_value = searcher.Q[node] if node.answer else None 
        n_value = searcher.N[node.answer] if node.answer else None
        info['children'] = children
        info['score'] = score
        info['Q'] = q_value
        info['N'] = n_value
        traces[node.id] = info 
        
    return traces 


def crisp_reason(model, reward, question, args):
    args.direct_sample_nums = args.roll_num // 2
    args.crisp_sample_nums = args.roll_num // (2 * args.beam_width * (args.max_depth_allowed - 1))
    generator = Generator(model, reward, args)
    searcher = CRISP_Searcher(
        topk=args.beam_width,
        max_depth=args.max_depth_allowed,
        ablation=args.ablation,
        max_cls=args.max_cls
    )

    root_node = CRISP_Node(
        parent=None,
        depth=0,
        question=question,
        generator=generator
    )

    pred = searcher.do_rollout(root_node)
    responses = [node.content for node in searcher.nodes]
    answers = [node.answer for node in searcher.nodes]
    traces = print_tree_from_root(searcher=searcher)

    return responses, answers, pred, traces