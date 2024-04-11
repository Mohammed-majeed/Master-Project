from ctransformers import AutoModelForCausalLM

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
                <Sequence>
                    <Condition>path_clear</Condition>
                    <Action>avoid_obstacle</Action>
                </Sequence>
            </Sequence>
            <Action>wander</Action>
        </Selector>
        <Selector>
            <Sequence>
                <Condition>target_reached</Condition>
                <Selector>
                    <Sequence>
                        <Condition>looking_for_nest</Condition>
                        <Action>return_to_nest</Action>
                    </Sequence>
                    <Sequence>
                        <Condition>path_clear</Condition>
                        <Action>avoid_obstacle</Action>
                    </Sequence>
                </Selector>
            </Sequence>
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

BT_3= """
<BehaviorTree>
    <Action>change_color</Action>
</BehaviorTree>
"""


# [INST] generate a behavior tree to avoid obstical and form line
# RESPONSE: {BT_2}[/INST]


read_examples= f"""

<s>[INST] generate a behavior tree to find the target then back to the home [/INST]
RESPONSE: {BT_1}</s>

<s>[INST] generate a behavior tree to change color [/INST]
RESPONSE: {BT_3}</s>
"""

prompt_template = f"""
{read_examples}
<s>[INST] Use only the following behaviors:['avoid_obstacle': 'Action node: avoid obstacle', 'change_color': 'Action node: Change color based on target detection', 'form_line': 'Action node: Form a line towards the center of the window', 'looking_for_nest': 'Action node: Look for the nest', 'nothing': 'Action node: does nothing', 'path_clear': 'Condition node: Check if the path is clear of obstacles', 'return_to_nest': 'Action node: Return to the nest', 'search_for_target': 'Action node: use this function to Search for target', 'target_detected': 'Condition node: Check if the target is detected', 'target_reached': 'Condition and/or Action node: Check if the target is reached', 'wander': 'Action node: Implement wandering behavior'] to construct behavior tree in XML format. User command:"find food"[/INST]  
Response: 
"""

def generate_BT(prompt,file_name):
    llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7b-Chat-GGUF",
                                                model_file="/home/mohammed/Desktop/llama-2-7b-chat.Q8_0.gguf",
                                                model_type="llama",
                                                context_length = 1024,                                            
                                                gpu_layers=16)
    print(llm(prompt_template))

generate_BT("find food", "behavior_tree_1.xml")
