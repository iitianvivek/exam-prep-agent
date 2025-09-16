import json
import os

sciq_infile = os.path.join(os.path.dirname(__file__), '../data/sciq_formatted.jsonl')
squad_infile = os.path.join(os.path.dirname(__file__), '../data/squad_formatted.jsonl')
end_outfile = os.path.join(os.path.dirname(__file__), '../data/enddataset.jsonl')

# Load all SciQ examples
sciq_examples = []
with open(sciq_infile, 'r', encoding='utf-8') as fin:
    for line in fin:
        ex = json.loads(line)
        sciq_examples.append(ex)

# Load 60,000 SQuAD examples
squad_examples = []
with open(squad_infile, 'r', encoding='utf-8') as fin:
    for i, line in enumerate(fin):
        if i >= 60000:
            break
        ex = json.loads(line)
        squad_examples.append(ex)

# Combine and write to enddataset.jsonl
with open(end_outfile, 'w', encoding='utf-8') as fout:
    for ex in sciq_examples + squad_examples:
        context = ex.get('context', '')
        qtype = ex.get('type', 'short')
        question = ex.get('question', '')
        answer = ex.get('answer', '')
        options = ex.get('options', None)
        if qtype == 'mcq' and options:
            prompt = f"Context: {context}\nType: mcq\nQuestion: {question}\nOptions: {options}\nAnswer: {answer}"
        else:
            prompt = f"Context: {context}\nType: short\nQuestion: {question}\nAnswer: {answer}"
        fout.write(json.dumps({'text': prompt}) + '\n')
print(f"Combined dataset written to {end_outfile}")
