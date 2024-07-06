qtype2prompt = {
    0: "provides a necessary assumption to",
    1: "provides a sufficient assumption to",
    2: "strengthens", 
    3: "weakens",
}

# NOTE: have removed the last period. 
qtype2definition = {
    0: "A premise provides a necessary assumption to the argument is that the premise must be true or is required in order for the argument to work",
    1: "A premise provides a sufficient assumption to the argument is that if the premise is added to the argument, it would make the argument logically valid",
    2: "A premise strengthens the argument is that the premise contains information that would strengthen an argument",
    3: "A premise weakens the argument is that the premise contains information that would weaken an argument",
}


template_ses = """You are an expert in logic. <prompt_definition_relation>. In the following, you are given an Argument and a Premise. Is the Premise <prompt_relation> the Argument? Please think step by step, and then answer "yes" or "no". 
"""


def get_system_prompts(qtype):
    system_prompt_ses = template_ses.replace('<prompt_definition_relation>', qtype2definition[qtype]).replace('<prompt_relation>', qtype2prompt[qtype])
    
    output = {
        "system_prompt_ses": system_prompt_ses,
    }

    return output
