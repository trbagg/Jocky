import re
import json
import random

SUBSTITUTION_LIST = [    
                        ['yeah', 'yerp', 'yea', 'yep', 'yes', 'ye', 'mhm'],
                        ['nope', 'nop', 'no'],
                        ['teto', 'kasane teto', 'red miku'],
                        ['miku', 'hatsune miku', 'blue teto'],
                        ['asf', 'as fuck'],
                        ['sex', 'seggs'],
                        ['prob', 'probably', 'prolly'],
                        ['idk', 'dont know', 'dunno'],
                        ['dumb', 'brain dead'],
                        ['fs', 'for sure'],
                        ['ts', 'type shit'],
                        ['cs', 'counter strike'],
                        ['smd', 'suck my dick', 'blow me', 'eat my dick'],
                        ['bitch', 'whore', 'slut', 'twerp'],
                        ['thx', 'thanks'],
                        ['r34', 'porn', 'nsfw'],
                        ['feet', 'barefoot', 'barefeet'],
                        ['np', 'no problem'],
                    ]

SUBSTITUTION_DICT = {
                        'league of legends': ['lol', 'league', 'riot ball torture'],
                        'ing': ['in'],
                        'hell': ['fuck'],
                        'sex': ['fucking', 'fuckin']
                    }

SUBSTITUTION_DICT.update({
                        item.lower(): [x.lower() for x in sublist if x != item] 
                        for sublist in SUBSTITUTION_LIST
                        for item in sublist
                    })

WILDCARD_PATTERN = r'\b(\w+)ing\b'

GIF_FAILED_PATTERN = r'\[gif:(?:(?!\]).)*$'

USER_ASSISTANT_STRUCTURE_PATTERN = r''

def main(input_path, output_path, iters = 3):

    print("Starting synthesization")
    synth(input_path=input_path, output_path=output_path, iters=iters)

def synth(input_path, output_path, iters):

    print(f"Synthesizing {input_path} to {output_path}.")

    with open(input_path, 'r') as f:
        raw_data = json.load(f)

    data = []

    for conversation in raw_data:
        data.append(conversation)
        prompt = conversation['prompt']
        chosen = conversation['chosen']
        rejected = conversation['rejected']

        turn = 'user'
        for p in prompt:
            if re.match(re.compile(GIF_FAILED_PATTERN), p['content']):
                raise ValueError("Bad gif bracket formatting at:", p['content'])
            elif re.match(re.compile(r'\[(?!(?:gif):)([^\]:]+): ?([^\]]+)\]'), p['content']):
                raise ValueError("Bad gif domain formatting at:", p['content'])
            elif '!ping' in p['content']:
                raise ValueError("Bad ping formatting at:", p['content'])
            elif p['role'] != turn:
                raise ValueError("Bad alternating pattern at:", p['content'])
            elif p['role'] == 'assistant' and p == prompt[-1]:
                raise ValueError("Bad prompt ending at:", p['content'])
            turn = ('user' if turn == 'assistant' else 'assistant')
            
        if chosen[0]['content'] == rejected[0]['content']:
            raise ValueError("Bad response formatting at:", f"{chosen[0]['content']} == {rejected[0]['content']}")
        for _ in range(iters):
            substitution = {"prompt": [{'role': p['role'], 'content': substitute(p['content'])} for p in prompt], "chosen": [{"role": "assistant", "content": substitute(chosen[0]['content'])}], "rejected": [{"role": "assistant", "content": substitute(rejected[0]['content'])}]}
            if substitution != conversation:
                data.append(substitution)
            else:
                pass


    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        print(f"Done. {len(data)} total results.")

def substitute(text):

    keys = sorted(SUBSTITUTION_DICT.keys(), key=len, reverse=True)
    pattern = re.compile(f'{r'\b(' + '|'.join(map(re.escape, keys)) + r')\b'}|{WILDCARD_PATTERN}', re.IGNORECASE)

    def apply_formatting(input, format):
        if format.islower():
            input = input.lower()
        if format.isupper():
            input = input.upper()
        if format.isupper():
            input = input[0].upper() + input[1:]
        return input

    def replace_match(match):
        if random.random() > 0.5:
            return match.group(0)
        if match.group(1):
            word = match.group(1).lower()
            options = SUBSTITUTION_DICT.get(word)
            return apply_formatting(random.choice(options), match.group(1)) if options else match.group(0)
        elif match.group(2):
            if match.group(0) == 'ping':
                return match.group(0)
            return apply_formatting(match.group(2) + 'in', match.group(2))

    return pattern.sub(replace_match, text)

if __name__ == '__main__':
    main('./content/handmade_orpo.json', './content/synth_orpo.json')