import torch
from transformers import pipeline


class NaiveModel:

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
The task is to evaluate the conclusion as 'True', 'False', or 'Uncertain' given the premises.
Answer the question directly! Simply respond with 'True', 'False', or 'Uncertain' between <EVALUATE> tags.

### Examples:
Example 1:
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

<EVALUATE>
Uncertain
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
True
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

<EVALUATE>
False
</EVALUATE>

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
Uncertain
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
False
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
True
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
        # max_new_tokens: int = 16,
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
