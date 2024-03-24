from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class LlamaAssistant:
    def __init__(self):
        self.model_name_or_path = "TheBloke/Llama-2-7B-Chat-GGML"
        self.model_basename = "llama-2-7b-chat.ggmlv3.q8_0.bin"
        self.model_path = None
        self.lcpp_llm = None

    def download_model(self):
        self.model_path = hf_hub_download(repo_id=self.model_name_or_path, filename=self.model_basename)

    def initialize_llama_model(self):
        # GPU
        self.lcpp_llm = Llama(
            model_path=self.model_path,
            n_threads=2,  # CPU cores
            n_batch=512,  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
            n_gpu_layers=32  # Change this value based on your model and your GPU VRAM pool.
        )

    def generate_behavior_tree(self, prompt):        

        if not self.lcpp_llm:
            print("Model not initialized. Please run 'initialize_llama_model()' first.")
            return None
        
        behaviors = call_behaviors()

        BT_example = """
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
        sec_BT_example= """<Action>form_line</Action>"""

        instructions = f'''Here are an examples of how the behavior tree could be structured:\n Fitst example: {BT_example} \n second example: {sec_BT_example} but you do not have to do it exactly the same structure.\n Use only the following available behaviors: {", ".join(behaviors)}
        to generate a behavior tree in XML format for simulated swarm robots to the following user'''

        content = f'Generate a behavior tree in XML format for simulated swarm robots to "{prompt}"'


        prompt_template = f'''SYSTEM: You are a helpful, respectful, and honest AI assistant. Your task is to generate well-structured XML code for behavior trees based on the provided instructions.

        INSTRUCTIONS: {instructions}

        USER: {content}

        ASSISTANT: 
        '''

        response = self.lcpp_llm(prompt=prompt_template, max_tokens=512, temperature=0, top_p=0.95,
                                 repeat_penalty=1.2, top_k=150, echo=False)
        print(response["choices"][0]["text"])

        return response["choices"][0]["text"]




def call_behaviors():        
    from main import SwarmAgent
    class_obj = SwarmAgent
    function_names = [func for func in dir(class_obj) 
                    if callable(getattr(class_obj, func)) 
                    and not func.startswith("__")
                    and getattr(class_obj, func).__qualname__.startswith(class_obj.__name__ + ".")]    
    return function_names


def generate_BT(prompt,file_name="behavior_tree.xml"):
    assistant = LlamaAssistant()
    assistant.download_model()
    assistant.initialize_llama_model()
    generated_tree = assistant.generate_behavior_tree(prompt)
    save_behavior_tree_xml(generated_tree , file_name)
    return generated_tree


import xml.etree.ElementTree as ET
def save_behavior_tree_xml(data, file_path):
    behavior_tree_xml = extract_behavior_tree(data)
    root = ET.fromstring(behavior_tree_xml)
    tree = ET.ElementTree(root)
    tree.write(file_path, encoding="utf-8", xml_declaration=True)

def extract_behavior_tree(data):
    start_tag = "<BehaviorTree>"
    end_tag = "</Behavior"
    start_index = data.find(start_tag) + len(start_tag)
    end_index = data.find(end_tag)
    behavior_tree_xml = data[start_index:end_index].strip()
    return behavior_tree_xml



if __name__ == "__main__":


    prompt = "avoid obstacle"
    print(generate_BT(prompt,file_name="behavior_tree.xml"))
