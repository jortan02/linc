import torch
from transformers import pipeline
from nltk.sem.logic import LogicParser
from nltk.inference.prover9 import Prover9
from collections import Counter

# import os
# os.environ['PROVER9'] = '/uufs/chpc.utah.edu/common/home/u1283221/cs6964/linc/prover9/source/bin'

class LincModel:

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
The task is to translate each of the sentences into FOL expressions inside <EVALUATE> tags.
Expressions should be adhere to the format of the Python NLTK package logic module.

### Examples:

Example 1:
<SENTENCES>
All dispensable things are environment-friendly.
All woodware is dispensable.
All paper is woodware.
No good things are bad.
All environment-friendly things are good.
A worksheet is either paper or is environment-friendly.
A worksheet is not dispensable.
</SENTENCES>

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
FOL: ((Paper(worksheet) & -EnvironmentFriendly(worksheet)) | (-Paper(worksheet) & EnvironmentFriendly(worksheet)))
TEXT: A worksheet is not dispensable.
FOL: -Dispensable(worksheet)
</EVALUATE>

Example 2:
<SENTENCES>
A La Liga soccer team ranks higher than another if it receives more points.
If two La Liga soccer teams recieve the same points, the team which recieves more points from the games between the two teams ranks higher.
Real Madrid and Barcelona are both La Liga soccer teams.
In La Liga 2021-2022, Real Madrid recieves 86 points and Barcelon recieves 73 points.
In La Liga 2021-2022, Real Madrid and Barcelona both recieve 3 points from the games between them.
In La Liga 2021-2022, Real Madrid ranks higher than Barcelona.
</SENTENCES>

<EVALUATE>
TEXT: A La Liga soccer team ranks higher than another if it receives more points.
FOL: all x y. (LaLiga(x) & LaLiga(y) & MorePoints(x, y) -> HigherRank(x, y))
TEXT: If two La Liga soccer teams recieve the same points, the team which recieves more points from the games between the two teams ranks higher.
FOL: all x y. (LaLiga(x) & LaLiga(y) & -MorePoints(x, y) & -MorePoints(y, x) & MorePointsInGameBetween(x, y) -> HigherRank(x, y))
TEXT: Real Madrid and Barcelona are both La Liga soccer teams.
FOL: LaLiga(realMadrid) & LaLiga(barcelona)
TEXT: In La Liga 2021-2022, Real Madrid recieves 86 points and Barcelon recieves 73 points.
FOL: MorePoints(realMadrid, barcelona)
TEXT: In La Liga 2021-2022, Real Madrid and Barcelona both recieve 3 points from the games between them.
FOL: -MorePointsInGameBetween(realMadrid, barcelona) & -MorePointsInGameBetween(barcelona, realMadrid)
TEXT: In La Liga 2021-2022, Real Madrid ranks higher than Barcelona.
FOL: HigherRank(realMadrid, barcelona)
</EVALUATE>

Example 3:
<SENTENCES>
All athletes are good at sports.
All Olympic gold medal winners are good athletes.
No scientists are good at sports.
All Nobel laureates are scientists.
Amy is good at sports or Amy is an Olympic gold medal winner.
If Amy is not a Nobel laureate, then Amy is not an Olympic gold medal winner.
If Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate.
</SENTENCES>

<EVALUATE>
TEXT: All athletes are good at sports.
FOL: all x. (Athlete(x) -> GoodAtSports(x))
TEXT: All Olympic gold medal winners are good athletes.
FOL: all x. (OlympicGoldMedalWinner(x) -> Athlete(x))
TEXT: No scientists are good at sports.
FOL: all x. (Scientist(x) -> -GoodAtSports(x))
TEXT: All Nobel laureates are scientists.
FOL: all x. (NobelLaureate(x) -> Scientist(x))
TEXT: Amy is good at sports or Amy is an Olympic gold medal winner.
FOL: GoodAtSports(amy) | OlympicGoldMedalWinner(amy)
TEXT: If Amy is not a Nobel laureate, then Amy is not an Olympic gold medal winner.
FOL: -NobelLaureate(amy) -> -OlympicGoldMedalWinner(amy)
TEXT: If Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate.
FOL: -OlympicGoldMedalWinner(amy) -> NobelLaureate(amy)
</EVALUATE>

Example 4:
<SENTENCES>
All people who are respected by others are people who contribute to the country. 
If a person is respected by others, then he/she contributes to the country. 
All people available to have a visit without any fees are those respected by others. 
All Customers who once served in the army are available to have a visit without any fees. 
All people who once were sentenced for thief stayed in prison for some time. 
All people who once stayed in prison for some time have a bad record in the local state. 
James was either once sentenced for thief or stayed in prison for some time. 
James is either with a bad record in the local state or respected by others. 
James contributes to the country.
</SENTENCES>

<EVALUATE>
TEXT: All people who are respected by others are people who contribute to the country. 
FOL: all x. (Respected(x) -> ContributeToCountry(x))
TEXT: If a person is respected by others, then he/she contributes to the country. 
FOL: all x. (Respected(x) -> ContributeToCountry(x))
TEXT: All people available to have a visit without any fees are those respected by others. 
FOL: all x. (HaveVisitWithoutAnyFees(x) -> Respected(x))
TEXT: All Customers who once served in the army are available to have a visit without any fees. 
FOL: all x. (Army(x) -> HaveVisitWithoutAnyFees(x))
TEXT: All people who once were sentenced for thief stayed in prison for some time. 
FOL: all x. (Thief(x) -> Prison(x))
TEXT: All people who once stayed in prison for some time have a bad record in the local state. 
FOL: all x. (Prison(x) -> BadRecord(x))
TEXT: James was either once sentenced for thief or stayed in prison for some time. 
FOL: ((Thief(james) & -Prison(james)) | (-Thief(james) & Prison(james)))
TEXT: James is either with a bad record in the local state or respected by others. 
FOL: ((BadRecord(james) & -Respected(james)) | (-BadRecord(james) & Respected(james)))
TEXT: James contributes to the country.
FOL: ContributeToCountry(james)
</EVALUATE>

Example 5:
<SENTENCES>
No songs are visual.
All folk songs are songs.
All videos are visual.
All movies are videos.
All sci-fi movies are movies.
Inception is a sci-fi movie.
Mac is neither a folk song nor a sci-fi movie.
Inception is a folk song.
</SENTENCES>

<EVALUATE>
TEXT: No songs are visual.
FOL: all x. (Song(x) -> -Visual(x))
TEXT: All folk songs are songs.
FOL: all x. (FolkSong(x) -> Song(x))
TEXT: All videos are visual.
FOL: all x. (Video(x) -> Visual(x))
TEXT: All movies are videos.
FOL: all x. (Movie(x) -> Video(x))
TEXT: All sci-fi movies are movies.
FOL: all x. (ScifiMovie(x) -> Movie(x))
TEXT: Inception is a sci-fi movie.
FOL: ScifiMovie(inception)
TEXT: Mac is neither a folk song nor a sci-fi movie.
FOL: -FolkSong(mac) & -ScifiMovie(mac)
TEXT: Inception is a folk song.
FOL: FolkSong(inception)
</EVALUATE>

Example 6:
<SENTENCES>
Every chef can cook.
Some people who aren't chefs can cook.
People who cook can make scrambled eggs and pasta.
If someone can make cookies and muffins, they are a baker.
Bakers who can also make scrambled eggs can make a good breakfast.
Luke can make cookies, scrambled eggs, and muffins, but not pasta.
Luke can make a good breakfast.
</SENTENCES>

<EVALUATE>
TEXT: Every chef can cook.
FOL: all x. (Chef(x) -> Cook(x))
TEXT: Some people who aren't chefs can cook.
FOL: exists x. (-Chef(x) & Cook(x))
TEXT: People who cook can make scrambled eggs and pasta.
FOL: all x. (Cook(x) -> (MakeScrambledEggs(x) & MakePasta(x)))
TEXT: If someone can make cookies and muffins, they are a baker.
FOL: all x. (MakeCookies(x) & MakeMuffins(x) -> Baker(x))
TEXT: Bakers who can also make scrambled eggs can make a good breakfast.
FOL: all x. ((Baker(x) & MakeScrambledEggs(x)) -> MakeGoodBreakfast(x))
TEXT: Luke can make cookies, scrambled eggs, and muffins, but not pasta.
FOL: MakeCookies(luke) & MakeScrambledEggs(luke) & MakeMuffins(luke) & -MakePasta(luke)
TEXT: Luke can make a good breakfast.
FOL: MakeGoodBreakfast(luke)
</EVALUATE>

### Question:

Here are the following sentences you need to translate:
<SENTENCES>
{premises}
{conclusion}
</SENTENCES>

Start your response with "<EVALUATE>". For each sentence in SENTENCES, write "TEXT:" and the sentence and "FOL:" with the FOL translation of the sentence. End your answer with "</EVALUATE>".\
"""
            case _:
                raise ValueError(f'Prompt format "{prompt_format}" does not exist')

    def answer(
        self,
        premises_list: list[list[str]],
        conclusion_list: list[str],
        index_list: list[int],
        repetitions: int = 5,
        out_of_k: int = 5,
        max_new_tokens: int = 1024,
    ):
        chats = []
        for premises, conclusion in zip(premises_list, conclusion_list):
            prompt = self.prompt_format.format(premises=premises, conclusion=conclusion)
            chat = [{"role": "user", "content": prompt}]
            chats.append(chat)
        batch_output = self.generator(
            chats,
            max_new_tokens=max_new_tokens,
            num_return_sequences=repetitions * out_of_k,
            pad_token_id = self.generator.tokenizer.eos_token_id
        )
        b_generated_results = [
            {
                "index": index.item(),
                "responses": [
                    [
                        response["generated_text"][-1]["content"]
                        for response in k_responses
                    ]
                    for k_responses in [
                        rk_responses[index : index + out_of_k]
                        for index in range(0, len(rk_responses), out_of_k)
                    ]
                ],
            }
            for rk_responses, index in zip(batch_output, index_list)
        ]
        return b_generated_results

    # def parse(self, b_generated_results: list[list[str]]):
    #     b_proved_results = []
    #     for generated_result in b_generated_results:
    #         index = generated_result["index"]
    #         k_responses = generated_result["responses"]
    #         fol_results = {
    #             "index": index,
    #             "responses": [],
    #         }
    #         for response in k_responses:
    #             try:
    #                 fol_strings = self.extract_fol_strings(response["generated_text"])
    #                 fol_exps = self.convert_fol_exps(fol_strings)
    #                 result = self.prover.prove(fol_exps[-1], fol_exps[:-1])
    #                 fol_results["responses"].append(result)
    #             except Exception:
    #                 fol_results["responses"].append("Error")
    #         b_proved_results.append(fol_results)
    #     return b_proved_results

    @staticmethod
    def _extract_fol_strings(text: str):
        lines = text.split("\n")
        fol_lines = [line[5:] for line in lines if "FOL:" in line]
        return fol_lines

    @staticmethod
    def _convert_fol_exps(fol_strings_premises: list[str], fol_string_conclusion: str):
        tlp = LogicParser()
        fol_string_not_conclusion = f"-({fol_string_conclusion})"
        fol_exps_premises = [tlp.parse(fol_string) for fol_string in fol_strings_premises]
        fol_exp_conclusion = tlp.parse(fol_string_conclusion)
        fol_exp_not_conclusion = tlp.parse(fol_string_not_conclusion)
        return fol_exps_premises, fol_exp_conclusion, fol_exp_not_conclusion
    
    @staticmethod
    def _parse(text: str):
        try:
            fol_strings = LincModel._extract_fol_strings(text)
            fol_strings_premises = fol_strings[:-1]
            fol_string_conclusion = fol_strings[-1]
            fol_exps_premises, fol_exp_conclusion, fol_exp_not_conclusion = LincModel._convert_fol_exps(fol_strings_premises, fol_string_conclusion)
            prover = Prover9()
            result = prover.prove(fol_exp_conclusion, fol_exps_premises)
            notResult = prover.prove(fol_exp_not_conclusion, fol_exps_premises)
            if result == notResult:
                return "Uncertain"
            else:
                return "True" if result else "False"
        except Exception:
            return "Error"
        
    @staticmethod
    def parse(texts: list[str]):
        results = [LincModel._parse(text) for text in texts]
        counter = Counter(results)
        value, count = counter.most_common()[0]
        return value
        
