from typing import List, Optional
from pydantic import BaseModel, Field
from huggingface_hub import hf_hub_download
import xml.etree.ElementTree as ET

# Assuming llama_cpp is already correctly set up in your environment
from llama_cpp import Llama

class BehaviorTreeConfig(BaseModel):
    model_name_or_path: str = "TheBloke/Llama-2-7B-Chat-GGML"
    model_basename: str = "llama-2-7b-chat.ggmlv3.q8_0.bin"
    n_threads: int = 2
    n_batch: int = 512
    n_gpu_layers: int = 32
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 0.95
    repeat_penalty: float = 1.2
    top_k: int = 150

class PromptRequest(BaseModel):
    prompt: str
    behaviors: List[dict] = []

class BehaviorTreeResponse(BaseModel):
    xml_content: Optional[str] = None

class LlamaAssistant:
    def __init__(self, config: BehaviorTreeConfig):
        self.config = config
        self.model_path = hf_hub_download(repo_id=config.model_name_or_path, filename=config.model_basename)
        self.lcpp_llm = Llama(
            model_path=self.model_path,
            n_threads=config.n_threads, 
            n_batch=config.n_batch,
            n_gpu_layers=config.n_gpu_layers,
            
        )
        
    def generate_behavior_tree(self, request: PromptRequest) -> BehaviorTreeResponse:
        if not self.lcpp_llm:
            print("Model not initialized. Please run 'initialize_llama_model()' first.")
            return BehaviorTreeResponse()
        
        BT_example_1 = """
      
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
        
        prompt_template = '''
        SYSTEM: As a behavior tree  specialist create a well-structured XML representation of a behavior tree based on the given task and available behaviors.
        INSTRUCTIONS: Here are examples of behavior trees:
        1. Example 1:
        {}
        3. Example 2:
        {}
        Feel free to devise alternative structures inspired by these examples.
        Use these behaviors: {} to construct a behavior tree in XML format. The user has given the command "{}".
        ASSISTANT RESPONSE:
        '''.format(BT_example_1, BT_example_3, request.behaviors, request.prompt)

        response = self.lcpp_llm(prompt=prompt_template, 
                                 max_tokens=self.config.max_tokens, 
                                 temperature=self.config.temperature, 
                                 top_p=self.config.top_p, 
                                 repeat_penalty=self.config.repeat_penalty, 
                                 top_k=self.config.top_k, 
                                 echo=False,
                                                           
                                 )        
        print(response)

        xml_content = extract_behavior_tree(response["choices"][0]["text"])
        print(xml_content)
        
        return BehaviorTreeResponse(xml_content=xml_content)


def call_behaviors():        
    from main import SwarmAgent
    class_obj = SwarmAgent
    function_names_and_docstrings = {}
    for func_name in dir(class_obj):
        if callable(getattr(class_obj, func_name)) and not func_name.startswith("__") and not func_name.startswith("update") and getattr(class_obj, func_name).__qualname__.startswith(class_obj.__name__ + "."):
            func = getattr(class_obj, func_name)
            if func.__doc__:
                function_names_and_docstrings[func_name] = func.__doc__.strip()
            else:
                function_names_and_docstrings[func_name] = "No docstring found."
    for func_name, docstring in function_names_and_docstrings.items():
        print(f"{func_name} : {docstring}")

    # print(f"{function_names_and_docstrings =}")

    return [function_names_and_docstrings]
# Example usage
# call_behaviors()



# def call_behaviors():        
#     from main import SwarmAgent
#     class_obj = SwarmAgent
#     function_names = [func for func in dir(class_obj) 
#                     if callable(getattr(class_obj, func)) 
#                     and not func.startswith("__")
#                     and not func.startswith("update")
#                     and getattr(class_obj, func).__qualname__.startswith(class_obj.__name__ + ".")]
#     print(function_names)    
    
#     return function_names

def extract_behavior_tree(data):
    start_tag = "<BehaviorTree>"
    end_tag = "</BehaviorTree>"
    start_index = data.find(start_tag) #+ len(start_tag)
    end_index = data.find(end_tag) + len(end_tag)
    if start_index == -1 or end_index == -1:
        return ""
    behavior_tree_xml = data[start_index:end_index].strip()
    return behavior_tree_xml

def save_behavior_tree_xml(data: str, file_path: str):
    if not data:
        print("No valid XML content to save.")
        return
    root = ET.fromstring(data)
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)


def generate_BT(prompt, file_name):
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    config = BehaviorTreeConfig()

    # Split the prompt into smaller chunks
    prompt_chunks = [prompt[i:i + 512] for i in range(0, len(prompt), 512)]

    # Initialize llama assistant
    llama_assistant = LlamaAssistant(config=config)

    # Generate behavior trees for each prompt chunk
    xml_contents = []
    for chunk in prompt_chunks:
        prompt_request = PromptRequest(prompt=chunk, behaviors=call_behaviors())
        bt_response = llama_assistant.generate_behavior_tree(prompt_request)
        if bt_response.xml_content:
            xml_contents.append(bt_response.xml_content)

    # Combine XML contents from all chunks
    combined_xml_content = "\n".join(xml_contents)

    if combined_xml_content:
        save_behavior_tree_xml(combined_xml_content, file_name)
        print(f"Behavior tree saved to {file_name}")
    else:
        print("Failed to generate behavior tree.")




# def generate_BT(prompt,file_name):
#     if prompt=="":
#         raise 
#     config = BehaviorTreeConfig()

#     prompt_request = PromptRequest(
#         prompt="find the food and back to the nest",
#         behaviors=call_behaviors() #["wander", "avoid_obstacle", "follow_line", "change_color"]
#     )

#     llama_assistant = LlamaAssistant(config=config)
#     bt_response = llama_assistant.generate_behavior_tree(prompt_request)
#     print(bt_response)
#     if bt_response.xml_content:
#         save_behavior_tree_xml(bt_response.xml_content, file_name)
#         print(f"Behavior tree saved to {file_name}")
#     else:
#         print("Failed to generate behavior tree.")

if __name__ == "__main__":

    prompt = "wander and avoid obsticals"
    print(generate_BT(prompt,file_name="behavior_tree_1.xml"))


    # # Example usage
    # config = BehaviorTreeConfig()
    # llama_assistant = LlamaAssistant(config=config)
    # prompt_request = PromptRequest(
    #     prompt="find the food and back to the nest",
    #     behaviors=call_behaviors() #["wander", "avoid_obstacle", "follow_line", "change_color"]
    # )
    # bt_response = llama_assistant.generate_behavior_tree(prompt_request)
    # print(bt_response)
    # if bt_response.xml_content:
    #     save_behavior_tree_xml(bt_response.xml_content, "behavior_tree_1.xml")
    #     print(f"Behavior tree saved to 'behavior_tree_1.xml'")
    # else:
    #     print("Failed to generate behavior tree.")



# from huggingface_hub import hf_hub_download
# from llama_cpp import Llama

# class LlamaAssistant:
#     def __init__(self):
#         self.model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
#         self.model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
#         self.model_path = None
#         self.lcpp_llm = None

#     def download_model(self):
#         self.model_path = hf_hub_download(repo_id=self.model_name_or_path, filename=self.model_basename)

#     def initialize_llama_model(self):
#         # GPU
#         self.lcpp_llm = Llama(
#             model_path=self.model_path,
#             n_threads=2,  # CPU cores
#             n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
#             n_gpu_layers=32  # Change this value based on your model and your GPU VRAM pool.
#         )

#     def generate_behavior_tree(self, prompt):        

#         if not self.lcpp_llm:
#             print("Model not initialized. Please run 'initialize_llama_model()' first.")
#             return None
        
#         behaviors = call_behaviors()

#         BT_example = """
#         <BehaviorTree>
#             <Sequence>
#                 <Selector>
#                     <Sequence>
#                         <Condition>behavior</Condition>
#                         <Sequence>
#                             <Condition>behavior</Condition>
#                             <Action>behavior</Action>
#                         </Sequence>
#                     </Sequence>
#                     <Action>behavior</Action>
#                 </Selector>
#                 <Selector>
#                     <Sequence>
#                         <Condition>behavior</Condition>
#                         <Action>behavior</Action>
#                     </Sequence>
#                     <Action>behavior</Action>
#                 </Selector>
#                 <Action>behavior</Action>
#             </Sequence>
#         </BehaviorTree>
#         """
#         sec_BT_example= """<Action>form_line</Action>"""

#         instructions = f'''Here are an examples of how the behavior tree could be structured:\n Fitst example: {BT_example} \n second example: {sec_BT_example} but you do not have to do it exactly the same structure.\n Use only the following available behaviors: {", ".join(behaviors)}
#         to generate a behavior tree in XML format for simulated swarm robots to the following user'''

#         content = f'Generate a behavior tree in XML format for simulated swarm robots to "{prompt}"'


#         prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.

#         INSTRUCTIONS: {instructions}

#         USER: {content}

#         ASSISTANT: 
#         '''

#         response = self.lcpp_llm(prompt=prompt_template, max_tokens=512, temperature=0, top_p=0.95,
#                                  repeat_penalty=1.2, top_k=150, echo=False)
#         print(response["choices"][0]["text"])

#         return response["choices"][0]["text"]




# def call_behaviors():        
#     from main import SwarmAgent
#     class_obj = SwarmAgent
#     function_names = [func for func in dir(class_obj) 
#                     if callable(getattr(class_obj, func)) 
#                     and not func.startswith("__")
#                     and getattr(class_obj, func).__qualname__.startswith(class_obj.__name__ + ".")]    
#     return function_names


# def generate_BT(prompt,file_name="behavior_tree.xml"):
#     assistant = LlamaAssistant()
#     assistant.download_model()
#     assistant.initialize_llama_model()
#     generated_tree = assistant.generate_behavior_tree(prompt)
#     save_behavior_tree_xml(generated_tree , file_name)
#     return generated_tree


# import xml.etree.ElementTree as ET
# def save_behavior_tree_xml(data, file_path):
#     behavior_tree_xml = extract_behavior_tree(data)
#     root = ET.fromstring(behavior_tree_xml)
#     tree = ET.ElementTree(root)
#     tree.write(file_path, encoding="utf-8", xml_declaration=True)

# def extract_behavior_tree(data):
#     start_tag = "<BehaviorTree>"
#     end_tag = "</Behavior"
#     start_index = data.find(start_tag) + len(start_tag)
#     end_index = data.find(end_tag)
#     behavior_tree_xml = data[start_index:end_index].strip()
#     return behavior_tree_xml



# if __name__ == "__main__":


#     prompt = "change color"
#     print(generate_BT(prompt,file_name="behavior_tree_1.xml"))
