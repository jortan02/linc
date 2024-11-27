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
Answer the question directly! Between <EVALUATE> tags, begin with 'Let's think step by step,' try to reason using chain of thought, and end with an ANSWER: of 'True', 'False', or 'Uncertain'.

### Examples:

Example 1:
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

<EVALUATE>
Let's think step by step. We want to evaluate if a worksheet is not dispensable.
From premise 6, we know that a worksheet is either paper or is environment-friendly.
If it is paper, then from premise 3, a worksheet is woodware, and from premise 2, a worksheet is dispensable.
If it is environment-friendly, we know it is good from premise 5, but we know nothing about whether it is dispensable.
Therefore, we don't know if a worksheet is dispensible or not, so the statement is uncertain.
ANSWER: Uncertain
</EVALUATE>

Example 2:
<PREMISES>
A La Liga soccer team ranks higher than another if it receives more points.
If two La Liga soccer teams recieve the same points, the team which recieves more points from the games between the two teams ranks higher.
Real Madrid and Barcelona are both La Liga soccer teams.
In La Liga 2021-2022, Real Madrid recieves 86 points and Barcelon recieves 73 points.
In La Liga 2021-2022, Real Madrid and Barcelona both recieve 3 points from the games between them.
</PREMISES>
<CONCLUSION>
In La Liga 2021-2022, Real Madrid ranks higher than Barcelona.
</CONCLUSION>

<EVALUATE>
Let's think step by step. We want to evaluate if Real Madrid ranks higher than Barcelona in La Liga 2021-2022.
From premise 1, a La Liga soccer team ranks higher than another if it receives more points.
Real Madrid received 86 points and Barcelona received 73 points (premise 4).
Therefore, Real Madrid has more points than Barcelona.
From these premises, it follows that Real Madrid ranks higher than Barcelona because they received more points.
ANSWER: True
</EVALUATE>

Example 3:
<PREMISES>
All athletes are good at sports.
All Olympic gold medal winners are good athletes.
No scientists are good at sports.
All Nobel laureates are scientists.
Amy is good at sports or Amy is an Olympic gold medal winner.
If Amy is not a Nobel laureate, then Amy is not an Olympic gold medal winner.
</PREMISES>
<CONCLUSION>
If Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate.
</CONCLUSION>

Example 4:
<PREMISES>
All people who are respected by others are people who contribute to the country. 
If a person is respected by others, then he/she contributes to the country. 
All people available to have a visit without any fees are those respected by others. 
All Customers who once served in the army are available to have a visit without any fees. 
All people who once were sentenced for thief stayed in prison for some time. 
All people who once stayed in prison for some time have a bad record in the local state. 
James was either once sentenced for thief or stayed in prison for some time. 
James is either with a bad record in the local state or respected by others. 
</PREMISES>
<CONCLUSION>
James contributes to the country.
</CONCLUSION>

<EVALUATE>
Let's think step by step. We want to evaluate if James contributes to the country.
From premises 5 and 6, if James was once sentenced for theft, then he stayed in prison, which means he has a bad record.
From premise 6, if James stayed in prison, he has a bad record.
From premise 8, James is either with a bad record or respected by others. Since having a bad record and being respected by others are mutually exclusive and based on the premises related to James's imprisonment and bad record, we can't determine conclusively if James is respected by others. Consequently, we can't definitively conclude if James contributes to the country.
ANSWER: Uncertain
</EVALUATE>

Example 5:
<PREMISES>
No songs are visual.
All folk songs are songs.
All videos are visual.
All movies are videos.
All sci-fi movies are movies.
Inception is a sci-fi movie.
Mac is neither a folk song nor a sci-fi movie.
</PREMISES>
<CONCLUSION>
Inception is a folk song.
</CONCLUSION>

<EVALUATE>
Let's think step by step. We want to evaluate if Inception is a folk song.
Since Inception is a sci-fi movie (premise 6), and all sci-fi movies are movies (premise 5), then Inception is a movie.
All movies are videos (premise 4), so Inception is a video.
All videos are visual (premise 3), therefore, Inception is visual.
We know from premise 1 that no songs are visual. Since Inception is visual, it cannot be a song. Thus, it is impossible for Inception to be a folk song.
ANSWER: False
</EVALUATE>

Example 6:
<PREMISES>
Every chef can cook.
Some people who aren't chefs can cook.
People who cook can make scrambled eggs and pasta.
If someone can make cookies and muffins, they are a baker.
Bakers who can also make scrambled eggs can make a good breakfast.
Luke can make cookies, scrambled eggs, and muffins, but not pasta.
</PREMISES>
<CONCLUSION>
Luke can make a good breakfast.
</CONCLUSION>

<EVALUATE>
Let's think step by step. We want to evaluate if Luke can make a good breakfast.
Since Luke can make cookies and muffins, Luke is a baker (premise 4).
As Luke can also make scrambled eggs, Luke fulfills the conditions in premise 5 for being able to make a good breakfast.
Therefore, based on these premises, Luke can make a good breakfast.
ANSWER: True
</EVALUATE>

<EVALUATE>
ANSWER: True
</EVALUATE>

### Question:
<PREMISES>
{premises}
</PREMISES>
<CONCLUSION>
{conclusion}
</CONCLUSION>\
"""
            case _:
                raise ValueError(f'Prompt format "{prompt_format}" does not exist.')

    def answer(
        self,
        premises_list: list[list[str]],
        conclusion_list: list[str],
        index_list: list[int],
        repetitions: int = 5,
        # max_new_tokens: int = 256,
    ):
        chats = []
        for premises, conclusion in zip(premises_list, conclusion_list):
            prompt = self.prompt_format.format(premises=premises, conclusion=conclusion)
            chat = [{"role": "user", "content": prompt}]
            chats.append(chat)
        batch_output = self.generator(
            chats,
            # max_new_tokens=max_new_tokens,
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
