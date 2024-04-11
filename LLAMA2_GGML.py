from typing import List, Optional
from pydantic import BaseModel, Field
# from huggingface_hub import hf_hub_download
import xml.etree.ElementTree as ET
# from pydantic import ValidationError


# Assuming llama_cpp is already correctly set up in your environment
from llama_cpp import Llama

error_output= 0


class BehaviorTreeConfig(BaseModel):
    # model_name_or_path: str = "TheBloke/Llama-2-7B-Chat-GGML"
    # model_basename: str = "llama-2-7b-chat.ggmlv3.q8_0.bin"

    # model_name_or_path: str = "/home/mohammed/Desktop/llama-2-7b-chat-codeCherryPop.Q4_K_M.gguf"
    model_name_or_path: str = "/home/mohammed/Desktop/llama-2-7b-chat.Q8_0.gguf"

    n_threads: int = 2
    n_batch: int = 1024
    n_gpu_layers: int = 32
    max_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 0.95
    repeat_penalty: float = 1.2
    top_k: int = 150


class PromptRequest(BaseModel):
    prompt: str
    behaviors: List[dict] = []

# class BehaviorTreeResponse(BaseModel):
#     xml_content: str

class LlamaAssistant:
    def __init__(self, config: BehaviorTreeConfig):
        self.prompt_template = ""

        self.config = config
        # self.model_path = hf_hub_download(repo_id=config.model_name_or_path, 
        #                                   filename=config.model_basename)
        
        self.lcpp_llm = Llama(
            model_path=self.config.model_name_or_path,
            n_threads=config.n_threads, 
            n_batch=config.n_batch,
            n_gpu_layers=config.n_gpu_layers,
        )
        
    def generate_behavior_tree(self, request: PromptRequest,fix_mistake="") -> str:
        # global error_output
        # if not self.lcpp_llm:
        #     print("Model not initialized. Please run 'initialize_llama_model()' first.")
        #     return BehaviorTreeResponse()

        BT_1 = """
        <BehaviorTree>
            <Sequence>
                <Selector>
                    <Sequence>
                        <Condition>target_detected</Condition>
                        <Sequence>
                            <Condition>path_clear</Condition>
                            <Action>target_reached</Action>
                        </Sequence>
                        <!-- If path to target is not clear, avoid obstacles -->
                        <Sequence>
                            <Condition>path_clear</Condition>
                            <Action>avoid_obstacle</Action>
                        </Sequence>
                    </Sequence>
                    <!-- If no target is detected, explore -->
                    <Action>wander</Action>
                </Selector>
                <!-- Once the target is found, return to the nest -->
                <Selector>
                    <Sequence>
                        <Condition>target_reached</Condition>
                        <Selector>
                            <!-- Check if path to nest is clear -->
                            <Sequence>
                                <Condition>looking_for_nest</Condition>
                                <Action>return_to_nest</Action>
                            </Sequence>
                            <!-- If path to nest is not clear, avoid obstacles -->
                            <Sequence>
                                <Condition>path_clear</Condition>
                                <Action>avoid_obstacle</Action>
                            </Sequence>
                        </Selector>
                    </Sequence>
                    <!-- If target not yet reached, continue current behavior (handled in the first Selector) -->
                    <Action>wander</Action>
                </Selector>
            </Sequence>
        </BehaviorTree>
            """
        BT_2 = """
        <BehaviorTree>
            <Selector>
                <Sequence>
                    <Condition>obstacle_detected</Condition>
                    <Action>avoid_obstacle</Action>
                </Sequence>
                <Action>form_line</Action>
            </Selector>
        </BehaviorTree>
            """
        

        read_examples= f"""
        generate a behavior tree to find the target then back to the home
        RESPONSE: {BT_1}</s>

        generate a behavior tree to avoid obstical and form line
        RESPONSE: {BT_2}</s>
        """


        
        BT_example_1 = """      
        <BehaviorTree>
                <Selector>
                    <Sequence>
                        <Condition>behavior</Condition>
                        <Action>behavior</Action>
                    </Sequence>
                    <Action>behavior</Action>
                </Selector>
        </BehaviorTree>
        """
        BT_example_2= """
        <BehaviorTree>
            <Action>behavior</Action>
        </BehaviorTree>
        """

        BT_example_3 = """
        <BehaviorTree>
            <Sequence>
                <Selector>
                    <Sequence>
                        <Condition>behavior</Condition>
                        <Sequence>
                            <Condition>behavior</Condition>
                            <Action>behavior</Action>
                        </Sequence>
                    </Sequence>
                    <Action>behavior</Action>
                </Selector>
                <Selector>
                    <Sequence>
                        <Condition>behavior</Condition>
                        <Action>behavior</Action>
                    </Sequence>
                    <Action>behavior</Action>
                </Selector>
                <Action>behavior</Action>
            </Sequence>
        </BehaviorTree>
        """
        

        # prompt_template = f'''
        # SYSTEM: As a helpful, respectful, and honest AI assistant, your main task is to create well-structured XML code for behavior trees according to the instructions provided.
        # INSTRUCTIONS: You're given examples of behavior tree structures, like:\n{BT_example} \nor\n {BT_example_1}\n \nNote that while these examples can guide you, feel free to devise different structures as needed. Your objective is to use the following available behaviors: {", ".join(request.behaviors)}, to craft a behavior tree in XML format. This tree should cater to simulated swarm robots based on the user's command below.
        # USER COMMAND: "{request.prompt}"
        # ASSISTANT RESPONSE:
        # '''
        
        # prompt_template_2 = f'''
        # SYSTEM: create XML code for a behavior tree given these examples: 
        # \n{BT_example_1} \n {BT_example_3}\n
        # Devise different BT structures as needed.
        # Use the following "{request.behaviors}" to generate behavior trees in XML format. 
        # USER: "{request.prompt}"
        # RESPONSE:
        # '''
        
        # prompt_template = '''
        # SYSTEM: As a behavior tree  specialist create a well-structured XML representation of a behavior tree based on the given task and available behaviors.
        # INSTRUCTIONS: Here are examples of behavior trees:
        # 1. Example 1:
        # {}
        # 3. Example 2:
        # {}
        # Feel free to devise alternative structures inspired by these examples.
        # Use these behaviors: {} to construct a behavior tree in XML format. The user has given the command "{}".
        # ASSISTANT RESPONSE:
        # '''.format(BT_example_1, BT_example_3, request.behaviors, request.prompt)

        if fix_mistake == "":
            examples= f'''
             Example 1:
            {BT_example_3}
             Example 2:
            {BT_example_2}'''
        else:
            examples=f'''Example:{BT_example_1}'''            

        self.prompt_template = f'''
        {read_examples}
        Use only the following behaviors:{request.behaviors} to construct behavior tree in XML format. User command:"{request.prompt}"  
        Response 1: {fix_mistake} 
        '''
        # First, consider how to implement the user command according to the given behaviors. Once you have completed all your steps, provide the answer based on your intermediate progress.        RESPONSE:
        print("###### prompt template ######")

        print(self.prompt_template)

        response = self.lcpp_llm(prompt=self.prompt_template, 
                                 max_tokens=self.config.max_tokens, 
                                 temperature=self.config.temperature, 
                                 top_p=self.config.top_p, 
                                 repeat_penalty=self.config.repeat_penalty, 
                                 top_k=self.config.top_k, 
                                 echo=False,                                                           
                                 )        
        print("###### response ######")
        print(response)

        xml_content = extract_behavior_tree(response["choices"][0]["text"])

        print("###### xml_content ######")
        print(xml_content)

        # output = BehaviorTreeResponse(xml_content=xml_content)

        # try:
        #     print("###### output ######")
        #     output = BehaviorTreeResponse(xml_content=xml_content)
        #     print(output)
        #     return output
            
        # except ValidationError as e:
        #     if error_output == 4:
        #         print('could not generate proper xml')
        #         exit() 
        #     error_output += 1
        #     fix_mistake = f"""you generated wrong XML behavior tree structure {xml_content}.
        #     you must structure it like the example start with <BehaviorTree> tag end with </BehaviorTree>tag.
        #     RESPONSE 2: """
        #     llama_assistant.generate_behavior_tree(prompt_request, fix_mistake)

        return xml_content 



def call_behaviors():        
    from main import SwarmAgent
    class_obj = SwarmAgent
    function_names_and_docstrings = {}
    for func_name in dir(class_obj):
        if callable(getattr(class_obj, func_name)) and not func_name.startswith("__")\
              and not func_name.startswith("update")\
              and not func_name.startswith("obstacle")\
                and getattr(class_obj, func_name).__qualname__.startswith(class_obj.__name__ + "."):
            func = getattr(class_obj, func_name)
            if func.__doc__:
                function_names_and_docstrings[func_name] = func.__doc__.strip()
            else:
                function_names_and_docstrings[func_name] = "No docstring found."
    # for func_name, docstring in function_names_and_docstrings.items():
    #     print(f"{func_name} : {docstring}")

    # print(f"{function_names_and_docstrings =}")

    return [function_names_and_docstrings]


def extract_behavior_tree(data):
    global config, prompt_request, llama_assistant, error_output

    start_tag = "<BehaviorTree>"
    end_tag = "</Behavior"
    start_index = data.find(start_tag) + len(start_tag)
    end_index = data.find(end_tag) #+ len(end_tag)
    if start_index == -1 or end_index == -1:
        if error_output == 4:
            print('could not generate proper xml')
            exit()  
        error_output+=1
        fix_mistake = f"""The generated XML behavior tree structure is incorrect:{data}.
        Ensure that it adheres to the correct structure, starting with the "<BehaviorTree>" root tag and ending with the "</BehaviorTree>" closing root tag. User command:"{prompt_request.prompt}".
        Response 2:"""
        
        return llama_assistant.generate_behavior_tree(prompt_request, fix_mistake)
    
    behavior_tree_xml = data[start_index:end_index].strip()
    return behavior_tree_xml

def save_behavior_tree_xml(data: str, file_path: str):
    if not data:
        print("No valid XML content to save.")
        return
    root = ET.fromstring(data)
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)


config = None
prompt_request = None
llama_assistant = None


def generate_BT(prompt,file_name):
    global config, prompt_request, llama_assistant 

    if prompt=="":
        raise 
    config = BehaviorTreeConfig()

    prompt_request = PromptRequest(
        prompt= prompt,
        behaviors=call_behaviors() 
    )

    llama_assistant = LlamaAssistant(config=config)
    bt_response = llama_assistant.generate_behavior_tree(prompt_request)
    if bt_response:
        save_behavior_tree_xml(bt_response, file_name)
        print(f"Behavior tree saved to {file_name}")
    else:
        print("Failed to generate behavior tree.")

if __name__ == "__main__":

    # prompt = "change color"

    prompt = "find food"
    print(generate_BT(prompt,file_name="behavior_tree_1.xml"))


################

# from llama_cpp import Llama
# llm = Llama(
#       model_path="/home/mohammed/Desktop/llama-2-7b-chat-codeCherryPop.Q4_K_M.gguf",
#       # n_gpu_layers=-1, # Uncomment to use GPU acceleration
#       # seed=1337, # Uncomment to set a specific seed
#       # n_ctx=2048, # Uncomment to increase the context window
# )
# output = llm(
#       "Q: Name the planets in the solar system? A: ", # Prompt
#       max_tokens=32, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion
# print(output)

##############

#     # # Example usage
#     # config = BehaviorTreeConfig()
#     # llama_assistant = LlamaAssistant(config=config)
#     # prompt_request = PromptRequest(
#     #     prompt="find the food and back to the nest",
#     #     behaviors=call_behaviors() #["wander", "avoid_obstacle", "follow_line", "change_color"]
#     # )
#     # bt_response = llama_assistant.generate_behavior_tree(prompt_request)
#     # print(bt_response)
#     # if bt_response.xml_content:
#     #     save_behavior_tree_xml(bt_response.xml_content, "behavior_tree_1.xml")
#     #     print(f"Behavior tree saved to 'behavior_tree_1.xml'")
#     # else:
#     #     print("Failed to generate behavior tree.")



# # from huggingface_hub import hf_hub_download
# # from llama_cpp import Llama

# # class LlamaAssistant:
# #     def __init__(self):
# #         self.model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
# #         self.model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
# #         self.model_path = None
# #         self.lcpp_llm = None

# #     def download_model(self):
# #         self.model_path = hf_hub_download(repo_id=self.model_name_or_path, filename=self.model_basename)

# #     def initialize_llama_model(self):
# #         # GPU
# #         self.lcpp_llm = Llama(
# #             model_path=self.model_path,
# #             n_threads=2,  # CPU cores
# #             n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
# #             n_gpu_layers=32  # Change this value based on your model and your GPU VRAM pool.
# #         )

# #     def generate_behavior_tree(self, prompt):        

# #         if not self.lcpp_llm:
# #             print("Model not initialized. Please run 'initialize_llama_model()' first.")
# #             return None
        
# #         behaviors = call_behaviors()

# #         BT_example = """
# #         <BehaviorTree>
# #             <Sequence>
# #                 <Selector>
# #                     <Sequence>
# #                         <Condition>behavior</Condition>
# #                         <Sequence>
# #                             <Condition>behavior</Condition>
# #                             <Action>behavior</Action>
# #                         </Sequence>
# #                     </Sequence>
# #                     <Action>behavior</Action>
# #                 </Selector>
# #                 <Selector>
# #                     <Sequence>
# #                         <Condition>behavior</Condition>
# #                         <Action>behavior</Action>
# #                     </Sequence>
# #                     <Action>behavior</Action>
# #                 </Selector>
# #                 <Action>behavior</Action>
# #             </Sequence>
# #         </BehaviorTree>
# #         """
# #         sec_BT_example= """<Action>form_line</Action>"""

# #         instructions = f'''Here are an examples of how the behavior tree could be structured:\n Fitst example: {BT_example} \n second example: {sec_BT_example} but you do not have to do it exactly the same structure.\n Use only the following available behaviors: {", ".join(behaviors)}
# #         to generate a behavior tree in XML format for simulated swarm robots to the following user'''

# #         content = f'Generate a behavior tree in XML format for simulated swarm robots to "{prompt}"'


# #         prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.

# #         INSTRUCTIONS: {instructions}

# #         USER: {content}

# #         ASSISTANT: 
# #         '''

# #         response = self.lcpp_llm(prompt=prompt_template, max_tokens=512, temperature=0, top_p=0.95,
# #                                  repeat_penalty=1.2, top_k=150, echo=False)
# #         print(response["choices"][0]["text"])

# #         return response["choices"][0]["text"]




# # def call_behaviors():        
# #     from main import SwarmAgent
# #     class_obj = SwarmAgent
# #     function_names = [func for func in dir(class_obj) 
# #                     if callable(getattr(class_obj, func)) 
# #                     and not func.startswith("__")
# #                     and getattr(class_obj, func).__qualname__.startswith(class_obj.__name__ + ".")]    
# #     return function_names


# # def generate_BT(prompt,file_name="behavior_tree.xml"):
# #     assistant = LlamaAssistant()
# #     assistant.download_model()
# #     assistant.initialize_llama_model()
# #     generated_tree = assistant.generate_behavior_tree(prompt)
# #     save_behavior_tree_xml(generated_tree , file_name)
# #     return generated_tree


# # import xml.etree.ElementTree as ET
# # def save_behavior_tree_xml(data, file_path):
# #     behavior_tree_xml = extract_behavior_tree(data)
# #     root = ET.fromstring(behavior_tree_xml)
# #     tree = ET.ElementTree(root)
# #     tree.write(file_path, encoding="utf-8", xml_declaration=True)

# # def extract_behavior_tree(data):
# #     start_tag = "<BehaviorTree>"
# #     end_tag = "</Behavior"
# #     start_index = data.find(start_tag) + len(start_tag)
# #     end_index = data.find(end_tag)
# #     behavior_tree_xml = data[start_index:end_index].strip()
# #     return behavior_tree_xml



# # if __name__ == "__main__":


# #     prompt = "change color"
# #     print(generate_BT(prompt,file_name="behavior_tree_1.xml"))
