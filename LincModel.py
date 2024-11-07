import torch
from transformers import pipeline
from nltk.sem.logic import LogicParser
from nltk.inference.prover9 import Prover9



class LincModel:
    
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
The task is to translate each of the premises and conclusions into FOL expressions, so that the expressions can be evaluated by a theorem solver to determine whether the conclusion follows from the premises.
Expressions should be adhere to the format of the Python NLTK package logic module:

The following is an example of a problem:
<PREMISES>
All dispensable things are environment-friendly.
All woodware is dispensable.
All paper is woodware.
No good things are bad.
All environment-friendly things are good.
A worksheet is either paper or is environment-friendly.
</PREMISES>
<CONCLUSION>
A worksheet is not dispensable.
</CONCLUSION>

The problem is answered in this format:
<EVALUATE>
TEXT: All dispensable things are environment-friendly.
FOL: all x. (Dispensable(x) -> EnvironmentFriendly(x))
TEXT: All woodware is dispensable.
FOL: all x. (Woodware(x) -> Dispensable(x))
TEXT: All paper is woodware.
FOL: all x. (Paper(x) -> Woodware(x))
TEXT: No good things are bad.
FOL: all x. (Good(x) -> -Bad(x))
TEXT: All environment-friendly things are good.
FOL: all x. (EnvironmentFriendly(x) -> Good(x))
TEXT: A worksheet is either paper or is environment-friendly.
FOL: ((Paper(Worksheet) & -EnvironmentFriendly( Worksheet)) | (-Paper(Worksheet) & EnvironmentFriendly(Worksheet)))
TEXT: A worksheet is not dispensable.
FOL: -Dispensable(Worksheet)
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
                raise ValueError(f"Prompt format \"{prompt_format}\" does not exist")

    def extract_fol_strings(self, text: str):
        lines = text.split("\n")
        fol_lines = [line[5:] for line in lines if "FOL:" in line]
        return fol_lines

    def convert_fol_exps(self, fol_strings: list[str]):
        tlp = LogicParser()
        fol_exps = [tlp.parse(fol_string) for fol_string in fol_strings]
        return fol_exps

    def answer(self, premises_list: list[str], conclusion_list: list[str], k: int = 5, max_new_tokens: int = 256):
        results = []
        for premises, conclusion in zip(premises_list, conclusion_list):
            
                prompt = self.prompt_format.format(premises=premises, conclusion=conclusion)
                chat = [
                    {"role": "user", "content": prompt}
                ]
                responses = self.generator(chat, max_new_tokens=max_new_tokens, num_return_sequences=k)
                fol_results = []
                try:
                    for response in responses: # TODO: Fix (extract_fol_strings)
                        fol_strings = self.extract_fol_strings(response["generated_text"])
                        fol_exps = self.convert_fol_exps(fol_strings)
                        prover = Prover9()
                        result = prover.prove(fol_exps[-1], fol_exps[:-1])
                        fol_results.append(result)
                except Exception:
                    results.append(False)
                results.append(fol_results)

        return results

