import torch
from transformers import pipeline


class CotModel:

    def __init__(self, model_name: str, prompt_format: str = "FOLIO"):
        self.model_name = model_name
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        match prompt_format:
            case "FOLIO":
                self.prompt_format = """\
The following is a first-order logic (FOL) problem.
The problem is to determine whether the conclusion follows from the premises.
The premises are given in the form of a set of first-order logic sentences.
The conclusion is given in the form of a single first-order logic sentence.
The task is to translate each of the premises and conclusions into FOL expressions.
Answer the question directly! Simply begin with 'Let's think step by step,' try to reason using chain of thought, and end with an ANSWER: of 'True', 'False', or 'Uncertain' between <EVALUATE> tags.

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
Let's think step by step. We want to evaluate if a worksheet is not dispensable. From premise 6, we know that a worksheet is either paper or is environment-friendly. If it is paper, then from premise 3, a worksheet is woodware, and from premise 2, a worksheet is dispensable. If it is environment-friendly, we know it is good from premise 5, but we know nothing about whether it is dispensable. Therefore, we don't know if a worksheet is dispensible or not, so the statement is uncertain.
ANSWER: Uncertain
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
                raise ValueError(f'Prompt format "{prompt_format}" does not exist.')

    def answer(
        self,
        premises_list: list[list[str]],
        conclusion_list: list[str],
        index_list: list[int],
        repetitions: int = 5,
        max_new_tokens: int = 256,
    ):
        chats = []
        for premises, conclusion in zip(premises_list, conclusion_list):
            prompt = self.prompt_format.format(premises=premises, conclusion=conclusion)
            chat = [{"role": "user", "content": prompt}]
            chats.append(chat)
        batch_output = self.generator(
            chats,
            max_new_tokens=max_new_tokens,
            num_return_sequences=repetitions,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        b_generated_results = [
            {
                "index": index,
                "responses": [
                    response["generated_text"][-1]["content"]
                    for response in r_responses
                ],
            }
            for r_responses, index in zip(batch_output, index_list)
        ]
        return b_generated_results
