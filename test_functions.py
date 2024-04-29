from easyeditor import BaseEditor
from easyeditor import KNHyperParams, FTHyperParams, KETrainingHparams,\
    ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams, \
    SERACTrainingHparams, SERACHparams, IKEHyperParams, FTApiHyperParams, LoRAHyperParams, \
    GraceHyperParams, PMETHyperParams,MELOHyperParams, MALMENTrainingHparams, MALMENHyperParams
from easyeditor import ZsreDataset, CounterFactDataset, AtomicDataset
from easyeditor import EditTrainer
from easyeditor.models.ike import encode_ike_facts
from sentence_transformers import SentenceTransformer
import pandas as pd
import pdb

def test_SERAC_Mistral(prompts, ground_truth, target_new, subject):

    # Load the specific hyperparameters for the SERAC model using the 'mistral' configuration.
    hparams = SERACHparams.from_hparams('./hparams/SERAC/mistral-7b-instruct.yaml')
    editor = BaseEditor.from_hparams(hparams)

    # Perform the edit operation using the loaded editor and hyperparameters.
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    # Output the metrics to see how well the editing performed.
    print(metrics)

    # Optionally, you might want to use pdb to debug or inspect the model after editing.
    pdb.set_trace()

    return metrics, edited_model



def test_MEMIT_Mistral(prompts, ground_truth, target_new, subject, locality_inputs, portability_inputs):

    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/mistral-7b-instruct.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        subject=subject,
        locality_inputs=locality_inputs,
        portability_inputs=portability_inputs,
        keep_original_weight=True
    )

    pdb.set_trace()

    return metrics, edited_model

def test_MEND_Mistral(prompts, ground_truth, target_new, subject):

    hparams = MENDHyperParams.from_hparams('./hparams/MEND/mistral-7b-instruct.yaml')
    editor = BaseEditor.from_hparams(hparams)
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True
    )

    pdb.set_trace()

    return metrics, edited_model



def test_SERAC_llama3(prompts, ground_truth, target_new, subject):
    hparams = SERACHparams.from_hparams('./hparams/SERAC/llama3-8b-instruct.yaml')
    editor = BaseEditor.from_hparams(hparams)
    # Assuming you have some dataset and other parameters set up
    ds = ZsreDataset('./data/zsre_mend_eval.json')
    metrics, edited_model, _ = editor.edit_dataset(ds)
    # Debugging or detailed output
    print(metrics)

    pdb.set_trace()
    return metrics, edited_model

def test_MEMIT_llama3(prompts, ground_truth, target_new, subject):
    hparams = MEMITHyperParams.from_hparams('./hparams/MEMIT/llama3-8b-instruct.yaml')
    editor = BaseEditor.from_hparams(hparams)

    metrics, edited_model, _ = editor.edit(prompts=prompts, target_new=target_new, subject=subject)
    # Debugging or detailed output
    print(metrics)
    pdb.set_trace()
    return metrics, edited_model


def test_MEND_llama3(prompts, ground_truth, target_new, subject):
    # Assuming the hyperparameters for MEND using Llama3 are defined in a YAML file
    hparams = MENDHyperParams.from_hparams('./hparams/MEND/llama3-8b-instruct.yaml')
    editor = BaseEditor.from_hparams(hparams)
    
    # Execute the edit method
    metrics, edited_model, _ = editor.edit(
        prompts=prompts,
        ground_truth=ground_truth,
        target_new=target_new,
        keep_original_weight=True  # or False depending on your test scenario
    )
    
    # Output the results for review
    print(metrics)
    
    # Optionally add a breakpoint for debugging
    pdb.set_trace()
    
    return metrics, edited_model