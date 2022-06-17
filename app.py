import pandas as pd
from transformers import AutoTokenizer
from transformers import RobertaTokenizer, EncoderDecoderModel
import gradio as gr
import string
from utils import (get_metadata,
                    append_prefix,
                    append_suffix,
                    process_field)

metadata = "../dataset/metadata.csv"
channel_dict, function_dict_trigger, function_dict_action, field_mapping, valid_field, channel_to_function_dict = get_metadata(path=metadata)

tokenizer = RobertaTokenizer.from_pretrained("imamnurby/rob2rand_merged_w_prefix_c_fc_field")

model_oneshot = EncoderDecoderModel.from_pretrained("imamnurby/rob2rand_merged_w_prefix_c_fc_field")
model_interactive = EncoderDecoderModel.from_pretrained("imamnurby/rob2rand_merged_w_prefix_c_fc_interactive")


###
# INTERACTIVE GENERATION FUNCTIONS 
###
def return_same(input_desc):
    return input_desc

def update_dropdown_trig_ch(df_result):
    list_result = []
    answer = ''
    for ind in df_result.index:
        if str(df_result['No.'][ind]) != '':
            answer = str(df_result['No.'][ind])+ ' - '+ str(df_result['Trigger Channel'][ind])
            list_result.append(answer)
    return gr.Dropdown.update(choices=list_result)

def update_dropdown_trig_func(df_result):
    list_result = []
    answer = ''
    for ind in df_result.index:
        if str(df_result['No.'][ind]) != '':
            answer = str(df_result['No.'][ind])+ ' - '+ str(df_result['Trigger Function'][ind])
            list_result.append(answer)
    return gr.Dropdown.update(choices=list_result)

def update_dropdown_action_ch(df_result):
    list_result = []
    answer = ''
    for ind in df_result.index:
        if str(df_result['No.'][ind]) != '':
            answer = str(df_result['No.'][ind])+ ' - '+ str(df_result['Action Channel'][ind])
            list_result.append(answer)
    return gr.Dropdown.update(choices=list_result)

def update_dropdown_action_func(df_result):
    list_result = []
    answer = ''
    for ind in df_result.index:
        if str(df_result['No.'][ind]) != '':
            answer = str(df_result['No.'][ind])+ ' - '+ str(df_result['Action Function'][ind])
            list_result.append(answer)
    return gr.Dropdown.update(choices=list_result)

def set_trigger_ch(df_result, string_chosen):
    index_chosen = string_chosen[0:1]
    index_chosen = int(index_chosen)
    return gr.Textbox.update(value = df_result.iloc[index_chosen-1]["Trigger Channel"])

def set_trig_func(df_result, string_chosen):
    index_chosen = string_chosen[0:1]
    index_chosen = int(index_chosen)
    return gr.Textbox.update(value = df_result.iloc[index_chosen-1]["Trigger Function"])

def set_action_ch(df_result, string_chosen):
    index_chosen = string_chosen[0:1]
    index_chosen = int(index_chosen)
    return gr.Textbox.update(value = df_result.iloc[index_chosen-1]["Action Channel"])

def set_final_result(tf, df_result, string_chosen):
    index_chosen = string_chosen[0:1]
    index_chosen = int(index_chosen)
    af = df_result.iloc[index_chosen-1]["Action Function"]
    tf_field = field_mapping.get(tf, "()")
    tf = tf + tf_field
    af_field = field_mapping.get(af, "()")
    af = af + af_field
    df_dict = {"Trigger": [tf],
                "Action": [af]}
    return pd.DataFrame(df_dict)

def generate_preds_tc(input_desc, n_beams_interactive):
    count_arr = []
    decoded_preds=[]
    descriptions=[]
    if input_desc!='':
        desc = input_desc.lower()
        desc = append_prefix(desc=desc, 
                            prefix= "GENERATE TRIGGER CHANNEL <pf> ")
        
        input_ids = tokenizer.encode(desc, return_tensors='pt')
        
        preds = model_interactive.generate(input_ids,
                            max_length=200,
                            num_beams=n_beams_interactive,
                            num_return_sequences=n_beams_interactive,
                            early_stopping=True)
        count = 0 
        for item in preds:
            temp_pred = (tokenizer.decode(item, skip_special_tokens=True))
            if temp_pred in channel_dict.keys():
                count = count + 1
                count_arr.append(count)
                decoded_preds.append(temp_pred)
                temp_desc = channel_dict.get(temp_pred, "null")
                descriptions.append(temp_desc)
    
    df = {'No.':count_arr,
          'Trigger Channel': decoded_preds,
          'Description': descriptions}
    return pd.DataFrame(df)

def generate_preds_tf(input_desc, n_beams_interactive, selected_tc):
    count_arr = []
    decoded_preds=[]
    descriptions=[]
    if input_desc!='' and selected_tc!='':
        desc = input_desc.lower()
        desc = append_prefix(desc=desc, 
                            prefix="GENERATE TRIGGER FUNCTION <pf> ")
        
        desc = append_suffix(desc=desc,
                            suffix=f" <out> {selected_tc}")
        
        input_ids = tokenizer.encode(desc, return_tensors='pt')
        
        preds = model_interactive.generate(input_ids,
                            max_length=200,
                            num_beams=n_beams_interactive,
                            num_return_sequences=n_beams_interactive,
                            early_stopping=True)
        count = 0 
        for item in preds:
            temp_pred = (tokenizer.decode(item, skip_special_tokens=True))
            if temp_pred in function_dict_trigger.keys():
                temp_desc = function_dict_trigger.get(temp_pred, "null")
                if selected_tc in temp_pred:
                    count = count + 1
                    count_arr.append(count)
                    decoded_preds.append(temp_pred)
                    descriptions.append(temp_desc)
        
    df = {'No.': count_arr,
        'Trigger Function': decoded_preds,
        'Description': descriptions}
    return pd.DataFrame(df)

def generate_preds_ac(input_desc, n_beams_interactive, selected_tc, selected_tf):
    count_arr = []
    decoded_preds=[]
    descriptions=[]
    if input_desc!='' and selected_tf!='':
        desc = input_desc.lower()
        desc = append_prefix(desc=desc, 
                            prefix= "GENERATE ACTION CHANNEL <pf> ")
        
        desc = append_suffix(desc=desc,
                            suffix=f" <out> {selected_tc} {selected_tf}")
        
        input_ids = tokenizer.encode(desc, return_tensors='pt')
        
        preds = model_interactive.generate(input_ids,
                            max_length=200,
                            num_beams=n_beams_interactive,
                            num_return_sequences=n_beams_interactive,
                            early_stopping=True)
        count = 0 
        for item in preds:
            temp_pred = (tokenizer.decode(item, skip_special_tokens=True))
            if temp_pred in channel_dict.keys():
                count = count + 1
                count_arr.append(count)
                decoded_preds.append(temp_pred)
                temp_desc = channel_dict.get(temp_pred, "null")
                descriptions.append(temp_desc)
        
    df = {'No.':count_arr,
        'Action Channel': decoded_preds,
        'Description': descriptions}
    return pd.DataFrame(df)

def generate_preds_af(input_desc, n_beams_interactive, selected_tc, selected_tf, selected_ac):
    count_arr = []
    decoded_preds=[]
    descriptions=[]
    if input_desc!='' and selected_ac!='':
        desc = input_desc.lower()
        desc = append_prefix(desc=desc, 
                            prefix="GENERATE TRIGGER FUNCTION <pf> ")
        
        desc = append_suffix(desc=desc,
                            suffix=f" <out> {selected_tc} {selected_tf} {selected_ac}")
        
        input_ids = tokenizer.encode(desc, return_tensors='pt')
        
        preds = model_interactive.generate(input_ids,
                            max_length=200,
                            num_beams=n_beams_interactive,
                            num_return_sequences=n_beams_interactive,
                            early_stopping=True)
        count = 0 
        for item in preds:
            temp_pred = (tokenizer.decode(item, skip_special_tokens=True))
            if temp_pred in function_dict_action.keys():
                temp_desc = function_dict_action.get(temp_pred, "null")
                
                if selected_ac in temp_pred:
                    count = count + 1
                    count_arr.append(count)
                    decoded_preds.append(temp_pred)
                    descriptions.append(temp_desc)
    
    df = {'No.':count_arr,
          'Action Function': decoded_preds,
         'Description': descriptions}
    df = pd.DataFrame(df)
    df.index.names = ['Ranking']
    return df
###

###
# ONESHOT GENERATION FUNCTIONS 
###
def generate_oneshot(input_desc, n_beams_oneshot):
    trigger = []
    trigger_desc = []
    action = []
    action_desc = []
    if input_desc!='':
        desc = input_desc.lower()    
        prefix="GENERATE ON THE FIELD-LEVEL GRANULARITY <pf> "
        desc = append_prefix(desc=desc, 
                            prefix=prefix)

        input_ids = tokenizer.encode(desc, return_tensors='pt')
        
        # activate beam search and early_stopping
        preds = model_oneshot.generate(input_ids,
                            max_length=200,
                            num_beams=n_beams_oneshot,
                            num_return_sequences=n_beams_oneshot,
                            early_stopping=True)
        
        decoded_preds = []
        for item in preds:
            decoded_preds.append(tokenizer.decode(item, skip_special_tokens=True))

        for item in decoded_preds:
            invalid_field = False
            splitted_items = item.split("<sep>")
            processed = []
            if len(splitted_items)==6:
                for idx, subitem in enumerate(splitted_items):
                    if idx!=2 or idx!=4:
                        subitem = subitem.strip()
                    processed.append(subitem)
                assert(len(processed)==6)
                temp_tf = processed[1]
                temp_af = processed[4]
                
                temp_tf_field = process_field(processed[2])
                for field in temp_tf_field:
                    if field not in valid_field:
                        invalid_field = True
                        break
                if invalid_field:
                    continue
                temp_tf_field = "(" + ", ".join(temp_tf_field) + ")"
                
                temp_af_field = process_field(processed[-1])
                for field in temp_af_field:
                    if field not in valid_field:
                        invalid_field = True
                        break
                if invalid_field:
                    continue
                temp_af_field = "(" + ", ".join(temp_af_field) + ")"
                
                if temp_tf in function_dict_trigger.keys() and temp_af in function_dict_action.keys():
                    temp_tf_desc = function_dict_trigger.get(temp_tf)
                    temp_af_desc = function_dict_action.get(temp_af)
                    
                    temp_tf = temp_tf + temp_tf_field
                    temp_af = temp_af + temp_af_field

                    trigger.append(temp_tf)
                    trigger_desc.append(temp_tf_desc)
                
                    action.append(temp_af)
                    action_desc.append(temp_af_desc)
                    
    df = {"Trigger": trigger,
        "Action": action,
        "Trigger Description": trigger_desc,
        "Action Description": action_desc}
    return pd.DataFrame(df)
###

###
# DISCOVER FUNCTIONS 
###
def generate_channel(input_desc, n_beams_discover):
    trigger = []
    trigger_func = []
    trigger_desc = []
    action = []
    action_func = []
    action_desc = []
    if input_desc!='':
        desc = input_desc.lower()    
        prefix="GENERATE CHANNEL ONLY WITHOUT FUNCTION <pf> "
        desc = append_prefix(desc=desc, 
                            prefix=prefix)

        input_ids = tokenizer.encode(desc, return_tensors='pt')
        
        # activate beam search and early_stopping
        preds = model_oneshot.generate(input_ids,
                            max_length=200,
                            num_beams=n_beams_discover,
                            num_return_sequences=n_beams_discover,
                            early_stopping=True)
        
        decoded_preds = []
        for item in preds:
            decoded_preds.append(tokenizer.decode(item, skip_special_tokens=True))
        
        for item in decoded_preds:
            channels = item.split("<sep>")
            channels = [ch.strip() for ch in channels]
            if len(channels)==2:
                if channels[0] in channel_dict.keys() and channels[1] in channel_dict.keys() and channels[0] in channel_to_function_dict.keys() and channels[1] in channel_to_function_dict.keys():
                    temp_tc_desc = channel_dict.get(channels[0])
                    trigger_desc.append(temp_tc_desc)
                    trigger.append(channels[0])
                    trigger_func.append(channel_to_function_dict.get(channels[0]))
                    
                    temp_ac_desc = channel_dict.get(channels[1])
                    action_desc.append(temp_ac_desc)
                    action.append(channels[1])
                    action_func.append(channel_to_function_dict.get(channels[1]))

    df_trigger = pd.DataFrame({"Trigger": trigger,
        "Available Functions": trigger_func,
        "Trigger Description": trigger_desc})
    
    df_action = pd.DataFrame({"Action": action,
        "Available Functions": action_func,
        "Action Description": action_desc})
    
    df_trigger.drop_duplicates(inplace=True)
    df_action.drop_duplicates(inplace=True)

    return pd.DataFrame(df_trigger), pd.DataFrame(df_action)

###
# MAIN GRADIO APP
###
demo = gr.Blocks()
with demo:
    gr.Markdown("<h1><center>RecipeGen: an Automated Trigger Action Programs (TAPs) Generation Tool</center></h1>")
    # gr.Markdown("This demo allows you to generate TAPs using functionality description described in English. You can learn the working detail of our tool from our paper")
    gr.Markdown("<h3>What is TAP?</h3>")
    gr.Markdown("""
        TAPs or Trigger Action Programs are event-driven rules used to automate smart devices and/or internet services. TAPs are written in the form of "IF a **{trigger}** is
        satisfied then execute an **{action}**, where the **{trigger}** and the **{action}** correspond to API calls. TAPs have been used in various use cases, ranging from home monitoring 
        system to business workflow automation.
        """)
    gr.Markdown("<h3>What is RecipeGen?</h3>")
    gr.Markdown("""
        RecipeGen is a deep learning-based tool that can assist end-users to generate TAPs using natural language description. End-users can describe the functionality of the intended TAP, then RecipeGen
        will generate the TAP candidates based on the given description.
    """)
    gr.Markdown("<h3>Working Mode</h3>")
    gr.Markdown("""
        - Interactive: generate a TAP using step-by-step wizard
        - One Click: generate a TAP using one click button
        - Function Discovery: Discover relevant functionalities from channels with a similar functionalities
    """)
    with gr.Tabs():
        with gr.TabItem("Interactive"):
            gr.Markdown("<h3><center>Instructions for Interactive Mode</center></h3>")
            gr.Markdown("""1. There are 5 generation steps, i.e., generating trigger channel, trigger function, action channel, action function, and the final TAP.
                2. **[STEP 1]** Describe the functionality in the `Functionality Description` text box. Click `Generate Trigger Channel` button. The channel candidates and their descriptions will show up in the `Trigger Channel Results` table.
                3. **[STEP 2]** Select a trigger channel from the dropdown `Select the Trigger Channel`. Click `Generate Trigger Function` button. The function candidates and their descriptions will show up in the `Trigger Function Results` table.
                4. **[STEP 3]** Select a trigger function from the dropdown `Select the Trigger Function`. Click `Generate Action Channel` button. The channel candidates and their descriptions will show up in the `Action Channel Results` table.
                5. **[STEP 4]** Select an action channel from the dropdown `Select the Action Channel`. Click `Generate Action Function` button. The function candidates and their descriptions will show up in the `Action Function Results` table.
                6. **[STEP 5]** Select an action function  from the `Select the Action Function` to generate the final TAP.""")
            gr.Markdown(""" NOTE: You can control how many sequences are returned by tuning the `Beam Width` slider. Larger value will cause longer generation time.
            """)
            
            with gr.Box():
                with gr.Column():
                    dropdown_example = gr.Dropdown(type ="value",choices = ["Log to my spreadsheet if motion is detected in the living room","Notify me when someone open the front door", "Turn on my Philips lamp every sunset","Update my picture in Twitter when I change my profile picture in Facebook","Send and append to my note  when I create a new bookmark"], label = "Here are some sample functionality descriptions that you can try")
                    button_use_example = gr.Button("Try this sample")
                         
            with gr.Box():
                with gr.Column():
                    
                    gr.Markdown("<h4><center>Step 1: Generate Trigger Channels</center></h4>")
                    textbox_input = gr.Textbox(label="Functionality Description", placeholder="Describe the functionality here")
                    n_beams_interactive = gr.Slider(minimum=2, maximum=100, value=20, step=1, label="Beam Width")
                    button_generate_tc = gr.Button("Generate Trigger Channels")
                    
                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>Trigger Channel Results</center></h4>")
                    table_tc = gr.Dataframe(headers=["No.","Trigger Channel", "Description"], row_count=1)
            
            with gr.Box():
                with gr.Column():
                    
                    gr.Markdown("<h4><center>Step 2: Generate Trigger Functions</center></h4>")
                    dropdown_tc = gr.Dropdown(label="Select the Trigger Channel",type="value", choices=[''])
                    textbox_selected_tc = gr.Textbox(value="", visible=False)
                    button_generate_tf = gr.Button("Generate Trigger Functions")
                    
                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>Trigger Function Results</center></h4>")
                    table_tf = gr.Dataframe(headers=["No.","Trigger Function", "Description"], row_count=1)
                
            with gr.Box():
                with gr.Column():
                    
                    gr.Markdown("<h4><center>Step 3: Generate Action Channels</center></h4>")
                    dropdown_tf = gr.Dropdown(label="Select the Trigger Function",type="value", choices=[''])
                    textbox_selected_tf = gr.Textbox(value="", visible=False)
                    button_generate_ac = gr.Button("Generate Action Channels")
                    
                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>Action Channel Results</center></h4>")
                    table_ac = gr.Dataframe(headers=["No.","Action Channel", "Description"], row_count=1)
            
            with gr.Box():
                with gr.Column():
                    gr.Markdown("<h4><center>Step 4: Generate Action Functions</center></h4>")
                    dropdown_ac = gr.Dropdown(label="Select the Action Channel",type="value", choices=[''])
                    textbox_selected_ac = gr.Textbox(value="", visible=False)
                    
                    button_generate_af = gr.Button("Generate Action Functions")
                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>Action Function Results</center></h4>")
                    table_af = gr.Dataframe(headers=["No.","Action Function", "Description"], row_count=1)

            with gr.Box():
                with gr.Column():
                    gr.Markdown("<h4><center>Step 5: Generate the Final TAP</center></h4>")
                    dropdown_af = gr.Dropdown(label="Select the Action Function",type="value", choices=[''])
                    table_final = gr.Dataframe(headers=["Trigger","Action"], row_count=1)
        
            button_use_example.click(return_same, inputs=[dropdown_example], outputs=[textbox_input])
            button_generate_tc.click(generate_preds_tc, inputs=[textbox_input, n_beams_interactive], outputs=[table_tc])
            
            table_tc.change(fn=update_dropdown_trig_ch, inputs=[table_tc], outputs=[dropdown_tc])
            dropdown_tc.change(fn=set_trigger_ch, inputs=[table_tc,dropdown_tc], outputs=[textbox_selected_tc])
            button_generate_tf.click(generate_preds_tf, inputs=[textbox_input, n_beams_interactive, textbox_selected_tc], outputs=[table_tf])
            
            table_tf.change(fn=update_dropdown_trig_func, inputs=[table_tf], outputs=[dropdown_tf])
            dropdown_tf.change(fn=set_trig_func, inputs=[table_tf,dropdown_tf], outputs=[textbox_selected_tf])
            button_generate_ac.click(generate_preds_ac, inputs=[textbox_input, n_beams_interactive, textbox_selected_tc, textbox_selected_tf], outputs=[table_ac])
            
            table_ac.change(fn=update_dropdown_action_ch, inputs=[table_ac], outputs=[dropdown_ac])
            dropdown_ac.change(fn=set_action_ch, inputs=[table_ac,dropdown_ac], outputs=[textbox_selected_ac])
            button_generate_af.click(generate_preds_af, inputs=[textbox_input, n_beams_interactive, textbox_selected_tc, textbox_selected_tf, textbox_selected_ac], outputs=[table_af])
            
            table_af.change(fn=update_dropdown_action_func, inputs=[table_af], outputs=[dropdown_af])
            dropdown_af.change(fn=set_final_result, inputs=[textbox_selected_tf, table_af, dropdown_af], outputs=[table_final])

        with gr.TabItem("One Click"):
            gr.Markdown("<h3><center>Instructions for One Click Mode</center></h3>")
            gr.Markdown("""
                1. Describe the functionality in the `Functionality Description` text box.
                2. Click `Generate TAP` button. The TAP candidates will show up in the `TAP Results` table. The table consists of 4 columns: Trigger, Action, Trigger Description, and Action Description. You can scroll the table vertically.
                """)
            gr.Markdown(""" NOTE: You can control how many sequences are returned by tuning the `Beam Width` slider. Larger value will cause longer generation time.""")
            
            with gr.Box():
                with gr.Column():
                    gr.Markdown("You can try some description samples below:")
                    dropdown_example = gr.Dropdown(type ="value",choices = ["Log to my spreadsheet if motion is detected in the living room","Notify me when someone open the front door", "Turn on my Philips lamp every sunset","Update my picture in Twitter when I change my profile picture in Facebook","Send and append to my note  when I create a new bookmark"], label = "Here are some sample functionality descriptions that you can try")
                    button_use_example = gr.Button("Try this sample")
                    
            with gr.Box():
                with gr.Column():
                    textbox_input = gr.Textbox(label="Functionality Description", placeholder="Describe the functionality here")
                    n_beams_oneshot = gr.Slider(minimum=2, maximum=100, value=20, step=1, label="Beam Width")
                    button_generate_oneshot = gr.Button("Generate TAPs")
                    
                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>TAP Results</center></h4>")
                    table_oneshot = gr.Dataframe(headers=["Trigger", "Action", "Trigger Description",  "Action Description"], row_count=1)
                    
            button_use_example.click(return_same, inputs=[dropdown_example], outputs=[textbox_input])            
            button_generate_oneshot.click(generate_oneshot, inputs=[textbox_input, n_beams_oneshot], outputs=[table_oneshot])
        
        with gr.TabItem("Function Discovery"):
            gr.Markdown("<h3><center>Instructions for One-shot Mode</center></h3>")
            gr.Markdown("""
                1. Describe the functionality in the `Functionality Description` text box.
                2. Click `Discover Functioanlities` button. The table containing relevant trigger and action channels will show up. Each channel is accompanied by a list of available functionalities.
                """)
            gr.Markdown(""" NOTE: You can control how many sequences are returned by tuning the `Beam Width` slider. Larger value will cause longer generation time.""")
            
            with gr.Box():
                with gr.Column():
                    gr.Markdown("You can try some description samples below:")
                    dropdown_example = gr.Dropdown(type ="value",choices = ["Log to my spreadsheet if motion is detected in the living room","Notify me when someone open the front door", "Turn on my Philips lamp every sunset","Update my picture in Twitter when I change my profile picture in Facebook","Send and append to my note  when I create a new bookmark"], label = "Here are some sample functionality descriptions that you can try")
                    button_use_example = gr.Button("Try this sample")
                    
            with gr.Box():
                with gr.Column():
                    textbox_input = gr.Textbox(label="Functionality Description", placeholder="Describe the functionality here")
                    n_beams_discover = gr.Slider(minimum=2, maximum=100, value=20, step=1, label="Beam Width")
                    button_discover_function = gr.Button("Discover Functions!")
                    
                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>Relevant Trigger Channels</center></h4>")
                    table_discover_tc = gr.Dataframe(headers=["Trigger", "Available Functions", "Trigger Description"], row_count=1)

                    gr.Markdown("<br>")
                    gr.Markdown("<h4><center>Relevant Action Channels</center></h4>")
                    table_discover_ac = gr.Dataframe(headers=["Action", "Available Functions", "Action Description"], row_count=1)
        
            button_use_example.click(return_same, inputs=[dropdown_example], outputs=[textbox_input])            
            button_discover_function.click(generate_channel, inputs=[textbox_input, n_beams_discover], outputs=[table_discover_tc, table_discover_ac])

demo.launch(server_port=8333, server_name="0.0.0.0")