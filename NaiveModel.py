import torch
from transformers import pipeline

class NaiveModel:
    
    def __init__(self, model_name: str, prompt_format: str = "FOLIO"):
        self.model_name = model_name
        self.generator = pipeline("text-generation", model=self.model_name, device_map="auto", torch_dtype=torch.float16)
        match prompt_format:
            case "FOLIO":
                self.prompt_format = """\
The following is a first-order logic (FOL) problem.
The problem is to determine whether the conclusion follows from the premises.
The premises are given in the form of a set of first-order logic sentences.
The conclusion is given in the form of a single first-order logic sentence.
The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises.
Answer the question directly! Simply respond with 'True', 'False', or 'Uncertain' between <EVALUATE> tags

The following is an example of a problem:
<PREMISES>
All dispensable things are environment-friendly.
All woodware is dispensable.
All paper is woodware.
No good things are bad.
All environment-friendly things are good.
A worksheet is either paper or is environment friendly.
</PREMISES>
<CONCLUSION>
A worksheet is not dispensable.
</CONCLUSION>

The problem is answered in this format:
<EVALUATE>
Uncertain
</EVALUATE>

<PREMISES>
{premises}
</PREMISES>
<CONCLUSION>
{conclusion}
</CONCLUSION>

<EVALUATE>\
"""
            case _:
                raise ValueError(f"Prompt format \"{prompt_format}\" does not exist.")

    def answer(self, premises_list: list[str], conclusion_list: list[str], k: int = 5, max_new_tokens: int = 16):
        chats = []
        for premises, conclusion in zip(premises_list, conclusion_list):
            prompt = self.prompt_format.format(premises=premises, conclusion=conclusion)
            chat = [
                {"role": "user", "content": prompt}
            ]
            chats.append(chat)
        batch_output = self.generator(chats, max_new_tokens=max_new_tokens, num_return_sequences=k)
        generated_texts = [[response["generated_text"][-1]["content"] for response in k_responses] for k_responses in batch_output]
        return generated_texts
