import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import json
import os
import shutil
import fitz


# Set up offload directory for model offloading
offload_directory = os.path.join(os.path.dirname(__file__), '..', 'offload')
if os.path.exists(offload_directory):
    shutil.rmtree(offload_directory)
os.makedirs(offload_directory, exist_ok=True)

# --- 1. Load the Final Model Checkpoint ---



final_model_dir = os.path.join(os.path.dirname(__file__), '..', 'distilgpt2-exam-prep-generator')
try:
    model = AutoModelForCausalLM.from_pretrained(
        final_model_dir,
        device_map="auto",
        low_cpu_mem_usage=True,
        offload_folder=offload_directory
    )
except Exception as e:
    print(f"Failed to load with auto device map: {e}")
    print("Trying to load with CPU device mapping...")
    model = AutoModelForCausalLM.from_pretrained(
        final_model_dir,
        device_map="cpu",
        low_cpu_mem_usage=True
    )
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)


# --- 2. Function to Read PDF Text ---
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"


# --- 3. Define the Agent's Core Logic ---

# --- Planner Agent ---
def planner_agent(context: str, num_concepts: int) -> list:
    instruction = (
        f"Instruction: List the {num_concepts} most important concepts from the following text as a JSON array of strings."
    )
    prompt = f"{instruction}\nText: {context}\nOutput:"
    generation_args = {
        "max_new_tokens": 100,
        "return_full_text": False,
        "do_sample": False,
    }
    output = pipe(prompt, **generation_args)
    generated_text = output[0]['generated_text']
    # Try to extract JSON array
    try:
        json_part = generated_text[generated_text.find('['):generated_text.rfind(']')+1]
        parsed = json.loads(json_part)
        if isinstance(parsed, list):
            return [str(x) for x in parsed][:num_concepts]
        else:
            return []
    except Exception as e:
        # Fallback: extract top N keywords (very simple)
        import re
        words = re.findall(r'\b\w+\b', context.lower())
        from collections import Counter
        stopwords = set(['the','and','of','in','to','a','is','as','for','with','on','by','an','at','from','it','that','this','has','are','be','was','or','his','he','she','their','they','which','but','not','have','over','one','all','more','than','into','its','also','who','had','been','were','will','can','such','other','may','most','among','each','any','some','so','if','do','does','did','out','up','no','we','you','i'])
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        most_common = [w for w, _ in Counter(keywords).most_common(num_concepts)]
        return most_common

# --- Executor Agent ---
def executor_agent(context: str, concept: str, question_type: str):
    qtype = question_type.lower()
    if qtype == 'mcq':
        # Diverse few-shot MCQ examples
        few_shot = (
            # Biology
            "Generate a multiple-choice question and its answer about: mitochondria.\n"
            "Context: The mitochondria is the powerhouse of the cell.\n"
            "Question: What is the powerhouse of the cell?\n"
            "Options: Nucleus | Mitochondria | Ribosome | Golgi apparatus\n"
            "Answer: Mitochondria\n"
            "---\n"
            # History
            "Generate a multiple-choice question and its answer about: World War II.\n"
            "Context: World War II ended in 1945 after the surrender of Germany and Japan.\n"
            "Question: In which year did World War II end?\n"
            "Options: 1939 | 1942 | 1945 | 1950\n"
            "Answer: 1945\n"
            "---\n"
            # Mathematics
            "Generate a multiple-choice question and its answer about: Pythagorean theorem.\n"
            "Context: The Pythagorean theorem states that in a right triangle, a^2 + b^2 = c^2.\n"
            "Question: Which formula represents the Pythagorean theorem?\n"
            "Options: a^2 + b^2 = c^2 | a^2 - b^2 = c^2 | a^2 + b = c^2 | a + b^2 = c^2\n"
            "Answer: a^2 + b^2 = c^2\n"
            "---\n"
            # Literature
            "Generate a multiple-choice question and its answer about: Shakespeare.\n"
            "Context: William Shakespeare wrote the play 'Romeo and Juliet.'\n"
            "Question: Who wrote 'Romeo and Juliet'?\n"
            "Options: Charles Dickens | William Shakespeare | Jane Austen | Mark Twain\n"
            "Answer: William Shakespeare\n"
            "---\n"
            # Geography
            "Generate a multiple-choice question and its answer about: Mount Everest.\n"
            "Context: Mount Everest is the highest mountain in the world, located in the Himalayas.\n"
            "Question: Where is Mount Everest located?\n"
            "Options: Alps | Andes | Himalayas | Rockies\n"
            "Answer: Himalayas\n"
            "---\n"
        )
        def run_mcq_prompt():
            prompt = (
                few_shot +
                f"Generate a multiple-choice question and its answer about: {concept}.\n"
                f"Context: {context}\n"
                f"Question: {concept}\n"
                f"Options:"
            )
            output = pipe(prompt, max_new_tokens=128, do_sample=False)
            generated_text = output[0]['generated_text']
            # Robust parsing for MCQ
            question, answer, options = None, None, None
            lines = generated_text.split('\n')
            for line in lines:
                if line.strip().startswith('Question:'):
                    question = line.split('Question:',1)[-1].strip()
                elif line.strip().startswith('Options:'):
                    options = line.split('Options:',1)[-1].strip()
                elif line.strip().startswith('Answer:'):
                    answer = line.split('Answer:',1)[-1].strip()
            # Post-process options for MCQ
            options_list = None
            if options:
                import re
                raw_opts = re.split(r'\||,|\n', options)
                cleaned = []
                seen = set()
                for opt in raw_opts:
                    opt = re.sub(r'^[A-Da-d][\).\:]\s*', '', opt).strip()
                    if opt and opt.lower() not in seen:
                        cleaned.append(opt)
                        seen.add(opt.lower())
                while len(cleaned) < 4:
                    cleaned.append(f"Option {chr(65+len(cleaned))}")
                options_list = cleaned[:4]
            result = {"question": question, "answer": answer}
            if options_list:
                result["options"] = options_list
            return result

        # Retry logic: up to 3 attempts
        for attempt in range(3):
            result = run_mcq_prompt()
            # Check for quality: must have 4 options, non-empty answer, and options not all identical
            opts = result.get("options", [])
            ans = result.get("answer", "")
            if (
                opts and len(opts) == 4 and ans and
                len(set([o.lower() for o in opts])) > 2 and
                ans.lower() in [o.lower() for o in opts]
            ):
                return result
        # Fallback: default template
        fallback_opts = [f"{concept} Option {i+1}" for i in range(4)]
        return {
            "question": f"Which of the following relates to {concept}?",
            "answer": concept,
            "options": fallback_opts
        }
    elif qtype == 'short':
        prompt = (
            f"Generate a question and its answer about: {concept}.\n"
            f"Context: {context}\n"
            f"Question: {concept}\n"
        )
    else:
        return "Invalid question type. Please choose 'mcq' or 'short'."

    output = pipe(prompt, max_new_tokens=128, do_sample=False)
    generated_text = output[0]['generated_text']
    # Robust parsing for both types
    question, answer, options = None, None, None
    lines = generated_text.split('\n')
    for line in lines:
        if line.strip().startswith('Question:'):
            question = line.split('Question:',1)[-1].strip()
        elif line.strip().startswith('Options:'):
            options = line.split('Options:',1)[-1].strip()
        elif line.strip().startswith('Answer:'):
            answer = line.split('Answer:',1)[-1].strip()
    # Post-process options for MCQ
    options_list = None
    if options:
        import re
        raw_opts = re.split(r'\||,|\n', options)
        cleaned = []
        seen = set()
        for opt in raw_opts:
            opt = re.sub(r'^[A-Da-d][\).\:]\s*', '', opt).strip()
            if opt and opt.lower() not in seen:
                cleaned.append(opt)
                seen.add(opt.lower())
        while len(cleaned) < 4:
            cleaned.append(f"Option {chr(65+len(cleaned))}")
        options_list = cleaned[:4]
    result = {"question": question, "answer": answer}
    if options_list:
        result["options"] = options_list
    return result


# --- 4. User Interface (CLI) ---
if __name__ == "__main__":
    # --- Test with base distilgpt2 model ---
    # (Debug prints removed)
    print(" Welcome to the Exam Prep Question Generator!")
    print("You can provide a PDF file or paste text directly.")
    print("Type 'quit' when you're done.")
    print("-" * 50)

    while True:
        user_input = input("Enter the path to your PDF file or paste your notes here:\n> ")
        if user_input.lower() == 'quit':
            break

        context = ""
        if os.path.exists(user_input) and user_input.lower().endswith('.pdf'):
            print(f"📄 Reading PDF file: {user_input}")
            context = extract_text_from_pdf(user_input)
            if "Error" in context:
                print(context)
                continue
        elif os.path.exists(user_input) and user_input.lower().endswith('.txt'):
            print(f"📄 Reading text file: {user_input}")
            try:
                with open(user_input, 'r', encoding='utf-8') as f:
                    context = f.read()
            except Exception as e:
                print(f"Error reading text file: {e}")
                continue
        else:
            context = user_input

        question_type = input("Choose question type ('mcq' or 'short'):\n> ")
        if question_type.lower() == 'quit':
            break

        try:
            num_questions = int(input("How many questions do you want to generate?\n> "))
            if num_questions < 1:
                print("Please enter a positive integer.")
                continue
        except ValueError:
            print("Invalid number. Please enter a positive integer.")
            continue

        # --- Orchestrator: Plan first ---
        plan = planner_agent(context, num_questions)
        if not plan:
            print(" Could not extract concepts from the text. Try again with different input.")
            continue
        # Limit to the number of questions requested
        plan = plan[:num_questions]
        print(f"✅ Plan created. Will generate questions for: {', '.join(plan)}")

        for i, concept in enumerate(plan):
            print(f"\n Executing step {i+1}: Generating question about '{concept}'...")
            result = executor_agent(context, concept, question_type)

            print("\n" + "="*25 + f" RESULT {i+1} " + "="*25)
            if isinstance(result, dict):
                print(f"Question: {result.get('question', 'N/A')}")
                print(f"Answer: {result.get('answer', 'N/A')}")
                if 'options' in result:
                    print("Options:")
                    for idx, opt in enumerate(result['options']):
                        print(f"  {chr(65+idx)}. {opt}")
            else:
                print(result)
            print("="*58 + "\n")