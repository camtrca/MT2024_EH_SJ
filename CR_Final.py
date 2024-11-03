import os
import re
import logging
import argparse
import traceback
from datetime import datetime
import networkx as nx
from difflib import SequenceMatcher

import openai

# To run the file, place this file inside the folder of evaluation, and run example such as below:
# python evaluation/CR_Final.py --qtype cycle --mode easy   

# Set up the Azure OpenAI API configuration
openai.api_type = "set up your own key"
openai.azure_endpoint ="set up your own key"
openai.api_version = "set up your own key"
openai.api_key = "set up your own key"

# ChatGPT tuning.
CRmodel = "set up your own model"
CRtemperature=0.75

###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
# Section 1: Logging and parsing functions.

# For Parsing arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description='Graph reasoning script.')
    # Question type
    parser.add_argument('--qtype', 
                        choices=['cycle', 
                                 'connectivity',
                                 'flow',
                                 'gnn',
                                 'hamilton',
                                 'topology'], 
                        default='cycle',
                        help='Type of question/problem to solve.')
    # Difficulty mode
    parser.add_argument('--mode',
                        choices=['easy', 'medium', 'hard'],
                        default='easy',
                        help='Difficulty mode of the graphs.')
    
    args = parser.parse_args()
    return args

# Function to log the prompt and result
def log_prompt(QType, segment, prompt, result):
    """
    Logs a prompt and result in the appropriate log file.

    Parameters:
    QType (str): The type of question/problem to solve.
    segment (str): The segment of the reasoning process (e.g. proposer, verifier, reporter).
    prompt (str): The prompt given to the AI.
    result (str): The output from the AI.
    """
    logger = get_logger(QType, segment)
    logger.info(f"\n========== {segment} Prompt ==========\n{prompt}\n\n========== {segment} Result ==========\n{result}\n")

# Function to load prompts from files
def load_prompt(file_path, **kwargs):
    """
    Loads a prompt from a file and formats it with given keyword arguments.

    Parameters:
    file_path (str): The path to the file containing the prompt.
    **kwargs: Keyword arguments to be used in formatting the prompt.

    Returns:
    str: The formatted prompt.
    """
    with open(file_path, 'r') as file:
        prompt = file.read()
    return prompt.format(**kwargs)

def get_logger(QType, segment):
    """
    Returns a logger object for logging prompts and results.

    Parameters:
    QType (str): The type of question/problem to solve.
    segment (str): The segment of the reasoning process (e.g. proposer, verifier, reporter).

    Returns:
    logging.Logger: The logger object.
    """
    
    logger_name = f"{QType}_{segment}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Create a directory for logs if it doesn't exist
        log_dir = f'logs/{QType}'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create a time-sensitive log file name within the logs directory
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_name = os.path.join(log_dir, f"{segment}_{current_time}.log")

        handler = logging.FileHandler(log_file_name,encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
# Section 2: Proposer, Refined Rroposer, Verifier and Reporter functions.

def Proposer(partial_graph, hypothesis, QType, **kwargs):
    """
    Generates a proposition based on a given partial graph and hypothesis for a specified question type.

    Args:
        partial_graph (str): The representation of the partial graph.
        hypothesis (str): The hypothesis related to the graph.
        QType (str): The type of question, which determines the prompt and analysis context.

    Returns:
        str: The generated proposition as a string. Returns None if an error occurs.
    """
    try:
        # Construct the prompt file path based on QType
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(script_dir, '..',
                                    'prompts', 
                                    QType,
                                    'proposer_prompt.txt')
        proposer_prompt = load_prompt(prompt_file,
                                      partial_graph=partial_graph,
                                      hypothesis=hypothesis,
                                      QType=QType,
                                      **kwargs
                                    )
        response = openai.chat.completions.create(
            model=CRmodel,  
            messages=[
                {"role": "system", "content": "You are a knowledgeable assistant specialized in graph theory."},
                {"role": "user", "content": proposer_prompt}
            ],
            max_tokens=500,
            temperature=CRtemperature
        )
        
        result = response.choices[0].message.content.strip()
        log_prompt(QType, 'Proposer', proposer_prompt, result)
        return result
    
    except Exception as e:
        print(f"Error in generating proposition: {e}")
        return None

def Refined_Proposer(partial_graph, hypothesis, last_failed_proposition, QType, **kwargs):
    """
    Generates an improved proposition based on a given partial graph, hypothesis, and a failed proposition for a specified question type.

    Args:
        partial_graph (str): The representation of the partial graph.
        hypothesis (str): The hypothesis related to the graph.
        last_failed_proposition (str): The last failed proposition.
        QType (str): The type of question, which determines the prompt and analysis context.
        **kwargs: Additional keyword arguments for prompt formatting.

    Returns:
        str: The generated improved proposition as a string. Returns None if an error occurs.
    """
    try:
        qtype_data = question_types.get(QType)
        if not qtype_data:
            raise ValueError(f"Unsupported question type: {QType}")

        # Build the prompt template with placeholders
        re_proposer_prompt_template = """
        The previous proposition didn't fully meet the requirements due to one or more of the following reasons: 
        - The proposition mentions edges not in the premises.
        - The proposition does not include the correct definition of {property}.
        - The reasoning may not be logically sound.

        Here is the last failed proposition:
        "{last_failed_proposition}"

        Please provide a new proposition that includes:
        - Accurate statements of the node connections, matching the partial graph.
        - A clear and concise definition of {property}, if not already provided: "{standard_definition}"
        - Logical analysis and observations based on the given edges that could help identify the presence or absence of {property}.
        - {additional_instructions}

        Please ensure:
        - Use clear and concise language.
        - Do not include trailing commas in lists of nodes.
        - Ensure all node lists end with the last node number without extra punctuation.
        - **If you can logically conclude that {positive_conclusion} or {negative_conclusion} based on the given edges, you may state this conclusion and provide your reasoning.**
        - Focus on providing a logically sound proposition based on the partial graph and hypothesis.
        - Avoid logical fallacies and ensure that your reasoning is valid and sound.
        - Do not include the previous failed proposition in the new proposition.

        Partial Graph:
        {partial_graph}

        Hypothesis:
        {hypothesis}
        """

        # Combine qtype_data and kwargs into a single dictionary for formatting
        prompt_variables = qtype_data.copy()
        prompt_variables.update({
            'partial_graph': partial_graph,
            'hypothesis': hypothesis,
            'last_failed_proposition': last_failed_proposition
        })
        prompt_variables.update(kwargs)  # Add any additional variables

        # Format the prompt
        re_proposer_prompt = re_proposer_prompt_template.format(**prompt_variables)

        response = openai.chat.completions.create(
            model=CRmodel,
            messages=[
                {"role": "system", "content": "You are a careful and precise assistant specialized in graph theory."},
                {"role": "user", "content": re_proposer_prompt}
            ],
            max_tokens=500,
            temperature=CRtemperature
        )

        result = response.choices[0].message.content.strip()
        log_prompt(QType, 'Refined Proposer', re_proposer_prompt, result)

        return result
    
    except Exception as e:
        print(f"Error in generating improved proposition: {e}")
        return None

def Verifier(proposition, premises, QType):
    
    """
    Verifier checks if the proposition is logically sound based on the given premises.

    Parameters
    ----------
    proposition : str
        The proposition generated by the Proposer.
    premises : dict
        The given partial graph.
    QType : str
        The type of question.

    Returns
    -------
    result : str
        'True' if the proposition is logically sound, 'False' otherwise.
    error_message : str
        An error message if the proposition is not logically sound.

    Raises
    ------
    ValueError
        If the question type is not supported.
    Exception
        If there is an error in parsing the proposition.
    """
    try:
        # Load question-specific data
        qtype_data = question_types.get(QType)
        if not qtype_data:
            raise ValueError(f"Unsupported question type: {QType}")

        # Attempt to parse the proposition
        edges_from_proposition = extract_edges_from_proposition(proposition)
        claimed_conclusion = infer_claim_from_proposition(proposition, qtype_data['conclusion_phrases'])
        
        if claimed_conclusion in ['present', 'absent']:
            error_message = "The proposition makes a final conclusion, which is not allowed at this stage."
            log_prompt(QType, 'Verifier', "Verifier Error", error_message)
            return 'False', error_message
        
    except Exception as e:
        # If parsing fails, consider the proposition invalid
        print(f"Parsing error: {e}")
        log_prompt(QType, 'Verifier', "Verifier Parsing Error", str(e))
        return 'False', "Parsing error"

    # Build the graph from premises
    G_premises = nx.Graph()
    G_premises.add_edges_from(premises['edges'])

    # Build the graph from the proposition (to validate consistency)
    G_proposition = nx.Graph()
    G_proposition.add_edges_from(edges_from_proposition)

    is_logically_sound = True
    error_messages = []

    # Normalize edges to ensure consistency in undirected graphs
    edges_in_proposition = normalize_edges(G_proposition.edges())
    edges_in_premises = normalize_edges(G_premises.edges())

    # Edge Consistency Check
    if not edges_in_proposition.issubset(edges_in_premises):
        is_logically_sound = False
        error_messages.append("The proposition mentions edges not in the premises.")

    # Definition Check
    definitions = extract_definitions(proposition, qtype_data['definition_keywords'])
    if not definitions:
        is_logically_sound = False
        error_messages.append(f"The proposition does not include a definition of {QType}.")
    else:
        standard_definition = qtype_data['standard_definition']
        definition_is_valid = any(is_similar_definition(defn, standard_definition) for defn in definitions)
        if not definition_is_valid:
            is_logically_sound = False
            error_messages.append(f"The proposition does not include the correct definition of {QType}.")

    # Logical Reasoning Check
    if claimed_conclusion in ['present', 'absent']:
        # Use the question-specific logical support check function
        logical_support_check_func_name = qtype_data['logical_support_check']
        logical_support_check = logical_support_functions[logical_support_check_func_name]
        if logical_support_check(G_premises, claimed_conclusion):
            pass  # Pass when the proposition is logically sound
        else:
            is_logically_sound = False
            error_messages.append("The proposition's conclusion is not logically supported by the given premises, it's likly that the conclusion made by the proposer contained direct analysis.")

    # Return the result
    if is_logically_sound:
        return 'True', None
    else:
        error_message = ' '.join(error_messages)
        log_prompt(QType, 'Verifier', "Verifier Error", error_message)
        return 'False', error_message

def Reporter(propositions, original_problem, QType):
    
    """
    Generates a comprehensive report based on accumulated propositions and a given problem.

    This function attempts to construct a logical conclusion regarding a specific question
    type in graph theory, using provided propositions and original problem statements. It 
    communicates with a chat model to analyze the input data and derive an answer.

    Args:
        propositions (list of tuples): A list containing propositions, where each proposition 
                                       is represented as a tuple. The first element of the tuple 
                                       is the proposition text.
        original_problem (str): The text describing the original problem that needs to be addressed.
        QType (str): The type of question being analyzed, used to determine the specific problem 
                     and context for the analysis.

    Returns:
        str: A conclusion derived from the propositions, starting with 'Yes' or 'No', followed by
             a concise reasoning. If no valid conclusion is reached after retries, an error message 
             is returned stating that no conclusion could be made.
    """
    # Construct the base prompt using the accumulated propositions
    base_propositions_text = "\n".join([f"Proposition {i+1}: {p[0]}" for i, p in enumerate(propositions)])

    # Load question-specific data
    qtype_data = question_types.get(QType)
    if not qtype_data:
        raise ValueError(f"Unsupported question type: {QType}")
    
    max_retries = 2  # Number of refinement attempts
    result = None  # Initialize result
    
    for attempt in range(max_retries):
        if attempt == 0:
            propositions_text = base_propositions_text
        else:
            # Add a note to the prompt to encourage the assistant to format the response correctly
            propositions_text = base_propositions_text + "\n\nNote: Please ensure your response starts with 'Yes' or 'No' and includes reasoning based on the propositions."
            print(f"Refinement needed in Reporter. Attempt {attempt + 1} of {max_retries}.")

        reporter_prompt = f"""
        You are an expert in graph theory. Based on the following propositions and analysis, determine whether {qtype_data['problem_statement']}.

        Original Problem:
        "{original_problem}"

        Accumulated Propositions:
        {propositions_text}

        The standard definition of the problem is:
        {qtype_data['standard_definition']}

        Please carefully analyze the propositions for any evidence that indicates the presence or absence of {qtype_data['property']}. Consider both possibilities equally and base your conclusion solely on the evidence provided.

        Can you conclude whether {qtype_data['question']}? Please answer with 'Yes' if {qtype_data['positive_answer']}, or 'No' if {qtype_data['negative_answer']}. Provide a brief explanation based on the propositions.

        Your response should start with 'Yes' or 'No', followed by your concise reasoning.
        """

        try:
            # Call the chat model to get the final decision
            response = openai.chat.completions.create(
                model=CRmodel, 
                messages=[
                    {"role": "system", "content": "You are a logical and analytical assistant."},
                    {"role": "user", "content": reporter_prompt}
                ],
                max_tokens=500,
                temperature=CRtemperature
            )

            result = response.choices[0].message.content.strip()
            log_prompt(QType, 'Reporter', reporter_prompt, result)

            # Check if the response starts with 'Yes' or 'No'
            if result.startswith('Yes') or result.startswith('No'):
                return result  # Accept the response if it meets the criteria
        except Exception as e:
            print(f"Error in Reporter: {e}")
            result = None  # Set result to None to handle in the final step

    # After retries, handle the failure case
    if result:
        print("Max retries reached in Reporter. Accepting the last response even if it doesn't start with 'Yes' or 'No'.")
        return result
    else:
        print("Error: No valid response obtained in Reporter after retries.")
        return "No conclusion could be made due to an error."

###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
# Section 3: The main cumulative_reasoning_flow function.

def cumulative_reasoning_flow(edges, n, hypothesis, QType, **kwargs):
    """
    A function that iteratively presents segments of a graph to a proposition generator,
    verifies the generated propositions using a verifier, and finally makes a conclusion
    using all verified propositions. The function takes in a list of edges, the number of
    nodes in the graph, and a hypothesis statement about the graph.

    Args:
        edges (list of tuples): A list of edges in the graph, where each edge is a tuple
                                of two nodes.
        n (int): The number of nodes in the graph.
        hypothesis (str): A hypothesis statement about the graph.

    Returns:
        list: A list of tuples, where each tuple contains the proposition text and the
              verification result ('True' or 'False').
        str: The final decision made by the reporter, either 'Yes' or 'No'.
    """
    # Sort edges by node before presenting
    sorted_edges = sort_edges_by_node(edges)

    verified_propositions = []
    iteration_count = 0
    presented_edges = []

    # Divide the sorted edges into segments for incremental presentation
    segment_size = 3  # Adjust as needed for difficulty level
    edge_segments = [sorted_edges[i:i + segment_size] for i in range(0, len(sorted_edges), segment_size)]

    # Store the original problem statement
    original_problem = f"Determine whether there is a {QType} in the graph with {n} nodes and the following edges: {edges}."

    # Iterate over each edge segment
    while iteration_count < len(edge_segments):
        # Add the current segment to the previously presented edges
        current_edges = edge_segments[iteration_count]
        presented_edges.extend(current_edges)

        # Incorporate previous propositions into the prompt
        previous_propositions_text = "\n".join([f"Proposition {i+1}: {p[0]}" for i, p in enumerate(verified_propositions)])

        # Build the 'Previous Propositions' section
        if previous_propositions_text:
            previous_propositions_section = f"Previous Propositions:\n{previous_propositions_text}"
        else:
            previous_propositions_section = ''

        # Construct the 'partial_graph' string without backslashes in expressions
        partial_graph = f"""
        
        In an undirected graph with {n} nodes and the following edges: {presented_edges}.
        {previous_propositions_section}

        Based on the given edges and previous propositions, please provide observations or insights that can help understand the graph's structure. **Do not make any final conclusions or decisions about the presence or absence of a {QType} at this stage. Focus on analyzing the relationships between nodes and any patterns that emerge.**

        Your response should:
        - Be purely observational and exploratory.
        - Avoid definitive statements about whether there is a {QType} in the graph.
        - Not include phrases like "Therefore, there is a {QType}" or "We can conclude that there is no {QType}."

        Remember, your goal is to build and combine upon the previous propositions and provide additional insights without reaching a final conclusion.
        """

        # Attempt to generate and verify the proposition
        max_retries = 3  # Set a limit for the number of retries
        valid_proposition = None
        last_failed_proposition = None

        for attempt in range(max_retries):
            # If retrying, include the failed prompt and request improvement
            if last_failed_proposition:
                proposition = Refined_Proposer(partial_graph, hypothesis, last_failed_proposition, QType, **kwargs)
            else:
                proposition = Proposer(partial_graph, hypothesis, QType, **kwargs)

            # Check if the proposition generation failed
            if proposition is None:
                print(f"Error: Proposition generation failed for graph with edges: {presented_edges}")
                # Return an empty proposition list and None as the final decision
                return [], None

            # Verifier: Verify the proposition
            premises = {'edges': presented_edges, 'n': n}
            verification_result, error_message = Verifier(proposition, premises, QType)

            # If the proposition is valid, break out of the retry loop
            if verification_result == 'True':
                valid_proposition = proposition
                break  # Exit the loop if a valid proposition is found
            else:
                # Log and store the failed proposition for retry
                last_failed_proposition = proposition
                print(f"\nInvalid proposition at iteration {iteration_count + 1}, retrying... (Attempt {attempt + 1} of {max_retries})")
                print(f"Verifier Error Messages: {error_message}\n")

        # After retries, if a valid proposition is found, append it to the list
        if valid_proposition:
            verified_propositions.append((valid_proposition, 'True'))
        else:
            # If no valid proposition was found after retries, append the last failed proposition
            print(f"\nAccepting the last proposition after {max_retries} attempts despite errors.")
            if last_failed_proposition:
                verified_propositions.append((last_failed_proposition, 'Accepted with errors: ' + error_message))
            else:
                # If no proposition was generated at all, return an empty list and None
                print(f"Error: No propositions generated for segment {current_edges}. Returning incomplete result.")
                return [], None

        # Print the current segment being presented
        print(f"Graph segment currently presenting: {current_edges}")
        print(f"Graph segment has been presented: {presented_edges}")
        iteration_count += 1

    # Reporter: Make the final decision based on all verified propositions
    if verified_propositions:
        final_decision = Reporter(verified_propositions, original_problem, QType)
        final_decision_lower = final_decision.lower()
        final_decision = "Yes" if "yes" in final_decision_lower else "No" if "no" in final_decision_lower else None
    else:
        # Default to None if no propositions were verified
        final_decision = None

    # Print the final decision
    print("#########Final Decision:###############")
    print(f"Final Decision for the current problem: {final_decision}")
    return verified_propositions, final_decision


###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
# Section 4: Logical Support Functions, for more question type, add more functions here as well as adding more to the dictionary.

def is_cycle_logically_supported(G, claimed_conclusion):
    """
    Checks if the claimed conclusion about cycles is logically supported.

    Parameters:
        G (networkx.Graph): The graph built from premises.
        claimed_conclusion (str): 'present', 'absent', or 'uncertain'.

    Returns:
        bool: True if the conclusion is logically supported, False otherwise.
    """
    has_cycle = len(nx.cycle_basis(G)) > 0

    if claimed_conclusion == 'present':
        return has_cycle
    elif claimed_conclusion == 'absent':
        return not has_cycle
    else:
        return True  # Accept if uncertain

def is_connectivity_logically_supported(G, claimed_conclusion):
    """
    Checks if the claimed conclusion about connectivity is logically supported.

    Parameters:
        G (networkx.Graph): The graph built from premises.
        claimed_conclusion (str): 'present' (connected), 'absent' (not connected), or 'uncertain'.

    Returns:
        bool: True if the conclusion is logically supported, False otherwise.
    """
    is_connected = nx.is_connected(G)

    if claimed_conclusion == 'present':
        return is_connected
    elif claimed_conclusion == 'absent':
        return not is_connected
    else:
        return True  # Accept if uncertain

def is_flow_logically_supported(G, claimed_conclusion, source, sink):
    """
    Checks if the claimed conclusion about maximum flow is logically supported.

    Parameters:
        G (networkx.DiGraph): The directed graph built from premises, with capacities as edge attributes.
        claimed_conclusion (str): The claimed maximum flow value (as a string).
        source (int or str): The source node.
        sink (int or str): The sink node.

    Returns:
        bool: True if the conclusion is logically supported, False otherwise.
    """
    # Calculate the maximum flow from source to sink
    try:
        flow_value, _ = nx.maximum_flow(G, source, sink, capacity='capacity')
    except nx.NetworkXError:
        # Flow calculation failed (e.g., source or sink not in graph)
        return False

    try:
        claimed_flow = float(claimed_conclusion)
    except ValueError:
        # Claimed conclusion is not a number
        return False

    # Allow for some numerical tolerance
    return abs(flow_value - claimed_flow) < 1e-6

def is_gnn_suitability_logically_supported(G, claimed_conclusion):
    """
    Checks if the claimed conclusion about GNN suitability is logically supported.

    Parameters:
        G (networkx.Graph): The graph built from premises, possibly with node and edge attributes.
        claimed_conclusion (str): 'present' (suitable), 'absent' (not suitable), or 'uncertain'.

    Returns:
        bool: True if the conclusion is logically supported, False otherwise.
    """
    # For simplicity, let's assume that a graph is suitable for GNN processing
    # if it has node features (embeddings) and there are no isolated nodes.

    has_node_features = all('embedding' in data for _, data in G.nodes(data=True))
    has_no_isolated_nodes = not list(nx.isolates(G))

    is_suitable = has_node_features and has_no_isolated_nodes

    if claimed_conclusion == 'present':
        return is_suitable
    elif claimed_conclusion == 'absent':
        return not is_suitable
    else:
        return True  # Accept if uncertain

def is_hamiltonian_logically_supported(G, claimed_conclusion):
    """
    Checks if the claimed conclusion about the existence of a Hamiltonian path is logically supported.

    Parameters:
        G (networkx.Graph): The graph built from premises.
        claimed_conclusion (str): 'present' (Hamiltonian path exists), 'absent' (does not exist), or 'uncertain'.

    Returns:
        bool: True if the conclusion is logically supported, False otherwise.
    """
    # Hamiltonian path checking is NP-complete; for small graphs, we can check all permutations
    from itertools import permutations

    nodes = list(G.nodes())
    n = len(nodes)

    has_hamiltonian_path = False

    for path in permutations(nodes):
        valid = True
        for i in range(n - 1):
            if not G.has_edge(path[i], path[i + 1]):
                valid = False
                break
        if valid:
            has_hamiltonian_path = True
            break  # Found a Hamiltonian path

    if claimed_conclusion == 'present':
        return has_hamiltonian_path
    elif claimed_conclusion == 'absent':
        return not has_hamiltonian_path
    else:
        return True  # Accept if uncertain

def is_topological_sort_possible(G, claimed_conclusion):
    """
    Checks if the claimed conclusion about the possibility of a topological sort is logically supported.

    Parameters:
        G (networkx.DiGraph): The directed graph built from premises.
        claimed_conclusion (str): 'present' (possible), 'absent' (not possible), or 'uncertain'.

    Returns:
        bool: True if the conclusion is logically supported, False otherwise.
    """
    try:
        # Attempt to get a topological sort
        list(nx.topological_sort(G))
        is_possible = True
    except nx.NetworkXUnfeasible:
        # Graph has a cycle; topological sort is not possible
        is_possible = False

    if claimed_conclusion == 'present':
        return is_possible
    elif claimed_conclusion == 'absent':
        return not is_possible
    else:
        return True  # Accept if uncertain

# Dictionary of logical support functions
logical_support_functions = {
    'is_cycle_logically_supported': is_cycle_logically_supported,
    'is_connectivity_logically_supported': is_connectivity_logically_supported,
    'is_flow_logically_supported': is_flow_logically_supported,
    'is_gnn_suitability_logically_supported': is_gnn_suitability_logically_supported,
    'is_hamiltonian_logically_supported': is_hamiltonian_logically_supported,
    'is_topological_sort_possible': is_topological_sort_possible,
}


# Dictionary of question types, each question type should contain all the necessary information, use cycle as a reference.

question_types = {
    'cycle': {
        'standard_definition': "A cycle is a path in a graph that starts and ends at the same node, passing through a sequence of distinct edges and nodes without revisiting any nodes, except the starting and ending node.",
        'definition_keywords': ['cycle'],
        'conclusion_phrases': {
            'present': [
                r"\bthere\s+is\s+a\s+cycle\b",
                r"\ba\s+cycle\s+exists\b",
                r"\bcycle\s+detected\b",
                r"\bthe\s+graph\s+contains\s+a\s+cycle\b",
                r"\bthe\s+graph\s+is\s+cyclic\b",
                r"\bfound\s+a\s+cycle\b",
                r"\bcycles\s+exist\s+in\s+the\s+graph\b",
                r"\bthis\s+forms\s+a\s+cycle\b",
                r"\bdefinitely\s+a\s+cycle\b",
            ],
            'absent': [
                r"\bthere\s+is\s+no\s+cycle\b",
                r"\bno\s+cycles\s+exist\b",
                r"\bthe\s+graph\s+is\s+acyclic\b",
                r"\bno\s+cycles\s+found\b",
                r"\bthe\s+graph\s+does\s+not\s+contain\s+a\s+cycle\b",
                r"\bcannot\s+find\s+a\s+cycle\b",
                r"\bthe\s+graph\s+lacks\s+cycles\b",
                r"\bdefinitely\s+no\s+cycle\b",
            ]
        },
        'logical_support_check': 'is_cycle_logically_supported', 
        'problem_statement': 'there is a cycle in the graph',
        'property': 'a cycle',
        'question': 'there is a cycle in the graph',
        'positive_answer': 'there is a cycle',
        'negative_answer': 'there is no cycle',
        'positive_conclusion': 'a cycle exists',
        'negative_conclusion': 'there is no cycle',
        'additional_instructions': 'Identify any sequences of nodes that could potentially form a cycle.\n- Discuss any closed paths where a node is reachable from itself via a sequence of edges.',
    },

    'connectivity': {
        'standard_definition': "A connected graph is a graph in which there is a path between every pair of vertices.",
        'definition_keywords': ['connected', 'connectivity'],
        'conclusion_phrases': {
            'present': [
                r"\bthe\s+graph\s+is\s+connected\b",
                r"\bthere\s+is\s+a\s+path\s+between\s+every\s+pair\s+of\s+nodes\b",
                r"\ball\s+nodes\s+are\s+reachable\s+from\s+each\s+other\b",
                r"\bthe\s+graph\s+has\s+connectivity\b",
                r"\bthe\s+graph\s+is\s+fully\s+connected\b",
                r"\bevery\s+node\s+can\s+reach\s+every\s+other\s+node\b",
            ],
            'absent': [
                r"\bthe\s+graph\s+is\s+not\s+connected\b",
                r"\bthere\s+is\s+no\s+path\s+between\s+some\s+pairs\s+of\s+nodes\b",
                r"\bthe\s+graph\s+is\s+disconnected\b",
                r"\bsome\s+nodes\s+are\s+not\s+reachable\s+from\s+others\b",
                r"\bthe\s+graph\s+lacks\s+connectivity\b",
                r"\bthe\s+graph\s+is\s+partially\s+connected\b",
            ]
        },
        'logical_support_check': 'is_connectivity_logically_supported',
        'problem_statement': 'the graph is connected',
        'property': 'connectivity',
        'question': 'Is the graph connected?',
        'positive_answer': 'the graph is connected',
        'negative_answer': 'the graph is not connected',
        'positive_conclusion': 'the graph is connected',
        'negative_conclusion': 'the graph is not connected',
        'additional_instructions': 'Analyze whether there is a path between every pair of nodes.\n- Discuss any disconnected components or isolated nodes.',
    },

    'flow': {
    'standard_definition': "In a flow network, a flow represents the amount of something passing through the network from a source to a sink, respecting capacity constraints.",
    'definition_keywords': ['flow', 'network flow', 'capacity', 'source', 'sink'],
    'conclusion_phrases': {
        'present': [
            r"\bthere\s+is\s+a\s+flow\s+from\s+source\s+to\s+sink\b",
            r"\ba\s+flow\s+exists\b",
            r"\bflow\s+is\s+possible\b",
            r"\bthe\s+network\s+allows\s+a\s+flow\b",
            r"\bthe\s+graph\s+supports\s+a\s+flow\b",
            r"\bflow\s+found\b",
        ],
        'absent': [
            r"\bthere\s+is\s+no\s+flow\s+from\s+source\s+to\s+sink\b",
            r"\bno\s+flow\s+exists\b",
            r"\bflow\s+is\s+not\s+possible\b",
            r"\bthe\s+network\s+does\s+not\s+allow\s+a\s+flow\b",
            r"\bthe\s+graph\s+does\s+not\s+support\s+a\s+flow\b",
            r"\bflow\s+not\s+found\b",
        ]
    },
        'logical_support_check': 'is_flow_logically_supported',
        'problem_statement': 'determine if there is a flow from the source to the sink',
        'property': 'flow',
        'question': 'Is there a flow from the source to the sink in the graph?',
        'positive_answer': 'there is a flow from the source to the sink',
        'negative_answer': 'there is no flow from the source to the sink',
        'positive_conclusion': 'a flow exists from the source to the sink',
        'negative_conclusion': 'there is no flow from the source to the sink',
        'additional_instructions': 'Assess whether a path exists from the source to the sink, considering any capacity constraints.\n- Discuss any bottlenecks or obstructions that may prevent flow.',
    },

    'gnn': {
    'standard_definition': "A Graph Neural Network (GNN) is a class of neural networks that operate on graph structures, leveraging the connections and features of nodes and edges.",
    'definition_keywords': ['GNN', 'graph neural network', 'node features', 'edge features'],
    'conclusion_phrases': {
        'present': [
            r"\bthe\s+graph\s+is\s+suitable\s+for\s+GNN\b",
            r"\bthe\s+graph\s+can\s+be\s+processed\s+by\s+a\s+GNN\b",
            r"\bGNN\s+applicable\s+to\s+this\s+graph\b",
            r"\bthe\s+graph\s+meets\s+GNN\s+requirements\b",
            r"\bthe\s+graph\s+is\s+compatible\s+with\s+GNNs\b",
        ],
        'absent': [
            r"\bthe\s+graph\s+is\s+not\s+suitable\s+for\s+GNN\b",
            r"\bthe\s+graph\s+cannot\s+be\s+processed\s+by\s+a\s+GNN\b",
            r"\bGNN\s+not\s+applicable\s+to\s+this\s+graph\b",
            r"\bthe\s+graph\s+does\s+not\s+meet\s+GNN\s+requirements\b",
            r"\bthe\s+graph\s+is\s+incompatible\s+with\s+GNNs\b",
        ]
    },
        'logical_support_check': 'is_gnn_suitability_logically_supported',
        'problem_statement': 'determine if the graph is suitable for GNN processing',
        'property': 'GNN suitability',
        'question': 'Is the graph suitable for processing with a Graph Neural Network?',
        'positive_answer': 'the graph is suitable for GNN processing',
        'negative_answer': 'the graph is not suitable for GNN processing',
        'positive_conclusion': 'the graph is compatible with GNNs',
        'negative_conclusion': 'the graph is incompatible with GNNs',
        'additional_instructions': 'Evaluate whether the graph has the necessary features (e.g., node and edge attributes) required for GNN processing.\n- Discuss how the graph structure may affect GNN performance.',
    },
    
    'hamilton': {
    'standard_definition': "A Hamiltonian cycle is a cycle in a graph that visits each vertex exactly once and returns to the starting vertex.",
    'definition_keywords': ['Hamiltonian', 'Hamiltonian cycle'],
    'conclusion_phrases': {
        'present': [
            r"\bthere\s+is\s+a\s+Hamiltonian\s+cycle\b",
            r"\ba\s+Hamiltonian\s+cycle\s+exists\b",
            r"\bHamiltonian\s+cycle\s+detected\b",
            r"\bthe\s+graph\s+contains\s+a\s+Hamiltonian\s+cycle\b",
            r"\bfound\s+a\s+Hamiltonian\s+cycle\b",
            r"\bthe\s+graph\s+is\s+Hamiltonian\b",
            r"\bthe\s+graph\s+has\s+a\s+Hamiltonian\s+cycle\b",
        ],
        'absent': [
            r"\bthere\s+is\s+no\s+Hamiltonian\s+cycle\b",
            r"\bno\s+Hamiltonian\s+cycles\s+exist\b",
            r"\bthe\s+graph\s+does\s+not\s+contain\s+a\s+Hamiltonian\s+cycle\b",
            r"\bcannot\s+find\s+a\s+Hamiltonian\s+cycle\b",
            r"\bthe\s+graph\s+is\s+not\s+Hamiltonian\b",
            r"\bthe\s+graph\s+lacks\s+a\s+Hamiltonian\s+cycle\b",
        ]
    },
        'logical_support_check': 'is_hamiltonian_logically_supported',
        'problem_statement': 'determine if there is a Hamiltonian cycle in the graph',
        'property': 'a Hamiltonian cycle',
        'question': 'Is there a Hamiltonian cycle in the graph?',
        'positive_answer': 'there is a Hamiltonian cycle',
        'negative_answer': 'there is no Hamiltonian cycle',
        'positive_conclusion': 'a Hamiltonian cycle exists',
        'negative_conclusion': 'there is no Hamiltonian cycle',
        'additional_instructions': 'Identify any cycles that visit every vertex exactly once.\n- Discuss possible sequences of nodes forming such a cycle.',
    },

    'topology': {
    'standard_definition': "A topology graph is a directed graph that defines dependencies between nodes, with each edge indicating a required visit order.",
    'definition_keywords': ['planar', 'planarity', 'topology'],
    'conclusion_phrases': {
        'present': [
            r"\bthe\s+graph\s+is\s+planar\b",
            r"\bthe\s+graph\s+can\s+be\s+drawn\s+without\s+edge\s+crossings\b",
            r"\bno\s+edges\s+cross\s+in\s+the\s+graph\b",
            r"\bthe\s+graph\s+can\s+be\s+embedded\s+in\s+the\s+plane\b",
            r"\bthe\s+graph\s+has\s+planarity\b",
            r"\bplanarity\s+confirmed\b",
        ],
        'absent': [
            r"\bthe\s+graph\s+is\s+not\s+planar\b",
            r"\bthe\s+graph\s+cannot\s+be\s+drawn\s+without\s+edge\s+crossings\b",
            r"\bedges\s+cross\s+in\s+the\s+graph\b",
            r"\bthe\s+graph\s+cannot\s+be\s+embedded\s+in\s+the\s+plane\b",
            r"\bthe\s+graph\s+lacks\s+planarity\b",
            r"\bplanarity\s+not\s+possible\b",
        ]
    },
        'logical_support_check': 'is_planarity_logically_supported',
        'problem_statement': 'determine if the graph is planar',
        'property': 'planarity',
        'question': 'Is the graph planar?',
        'positive_answer': 'the graph is planar',
        'negative_answer': 'the graph is not planar',
        'positive_conclusion': 'the graph is planar',
        'negative_conclusion': 'the graph is not planar',
        'additional_instructions': 'Determine if the graph can be drawn without edge crossings.\n- Discuss Kuratowski\'s theorem and the presence of K₅ or K₃,₃ subgraphs.',
    },

    # Add other question types as needed
    # '...':{}
}

###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
# Section 5: Helper functions.

def is_similar_definition(definition, standard_definition):
    """
    Compares the extracted definition with the standard definition using semantic similarity.
    
    Parameters:
        definition (str): The definition extracted from the proposition.
        standard_definition (str): The standard definition of a cycle.
    
    Returns:
        bool: True if the definitions are similar enough, False otherwise.
    """
    ratio = SequenceMatcher(None, definition.lower(), standard_definition.lower()).ratio()
    return ratio >= 0.6  # Adjust threshold as needed

def extract_definitions(proposition, definition_keywords):
    """
    Extracts definitions from the proposition based on definition keywords.

    Parameters:
        proposition (str): The proposition text.
        definition_keywords (list): Keywords to look for in definitions.

    Returns:
        List of strings containing definitions.
    """
    sentences = re.split(r'[.?!]\s*', proposition)
    definitions = []
    for sentence in sentences:
        if any(keyword in sentence.lower() for keyword in definition_keywords) and ('is' in sentence.lower() or 'refers to' in sentence.lower()):
            definitions.append(sentence.strip())
    return definitions

def normalize_edges(edges):
    """
    Normalize edges so that (u, v) and (v, u) are treated as the same undirected edge.
    
    Parameters:
        edges (list of tuples): A list of edges to normalize.
    
    Returns:
        set of tuples: A set of edges where each edge is represented as a sorted tuple.
    """
    return {tuple(sorted(edge)) for edge in edges}

def extract_edges_from_proposition(proposition):
    """
    Extracts edges from the proposition text.

    Parameters:
        proposition (str): The proposition text.

    Returns:
        List of tuples representing edges.
    """
    edges = []
    # Patterns to match different ways of describing edges
    patterns = [
        r"Node\s+(\d+)\s+(?:is connected to|connects to|connected with|has an edge with|is linked to|links to)\s+nodes?\s+([0-9,\s]+)(?:\.|$)",
        r"An edge exists between nodes?\s+(\d+)\s+and\s+(\d+)",
        r"Nodes?\s+(\d+)\s+and\s+(\d+)\s+are connected",
        r"Edge\s+between\s+nodes?\s+(\d+)\s+and\s+(\d+)",
        r"Edge\s+from\s+node\s+(\d+)\s+to\s+node\s+(\d+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, proposition, re.IGNORECASE)
        for match in matches:
            nodes = [int(n.strip()) for n in match if n.strip().isdigit()]
            if len(nodes) == 2:
                edges.append(tuple(nodes))
            elif len(nodes) > 2:
                # Handle cases where one node connects to multiple nodes
                node = nodes[0]
                for cn in nodes[1:]:
                    edges.append((node, cn))
    return edges

def infer_claim_from_proposition(proposition, conclusion_phrases):
    """
    Analyzes the proposition to identify whether it contains definitive statements about the presence or absence of a specific property.

    Parameters:
        proposition (str): A string containing the proposition to be analyzed.
        conclusion_phrases (dict): A dictionary with 'present' and 'absent' keys containing lists of regex patterns.

    Returns:
        str: A statement categorizing the proposition's claim as 'present', 'absent', or 'uncertain'.
    """
    proposition_lower = proposition.lower()

    # Words indicating uncertainty
    uncertainty_words = ['possible', 'possibility', 'potential', 'could', 'might', 'may', 'possibly', 'unlikely',
                         'perhaps', 'suggests', 'indicates', 'could be', 'cannot be sure', 'not certain']

    # Function to check for uncertainty words near a match
    def has_uncertainty_words(text, start, end):
        surrounding_text = text[max(0, start - 20):min(len(text), end + 20)]
        return any(word in surrounding_text for word in uncertainty_words)

    # Check for definitive claims of presence
    for phrase in conclusion_phrases.get('present', []):
        for match in re.finditer(phrase, proposition_lower):
            start, end = match.span()
            if has_uncertainty_words(proposition_lower, start, end):
                continue  # Skip this match due to uncertainty words
            else:
                return 'present'

    # Check for definitive claims of absence
    for phrase in conclusion_phrases.get('absent', []):
        for match in re.finditer(phrase, proposition_lower):
            start, end = match.span()
            if has_uncertainty_words(proposition_lower, start, end):
                continue  # Skip this match due to uncertainty words
            else:
                return 'absent'

    # Default to 'uncertain' if no definitive claims are found
    return 'uncertain'

# Predict function with integrated CR logic, returning reasoning steps and final decision
def predict(graph_edges_list, n, QType, hypothesis, **kwargs):
    """
    Runs the cumulative reasoning (CR) flow for a given list of graph edges and node count, 
    and collects the final decisions and reasoning steps for all graphs.

    Args:
        graph_edges_list (list): A list of graph edges, where each edge is a list of two node IDs.
        n (int): The number of nodes in the graph.
        QType (str): The type of question being asked, used to determine the specific problem and context.
        hypothesis (str): The hypothesis statement for the problem.
        **kwargs: Additional keyword arguments for prompt formatting.

    Returns:
        tuple: A tuple containing two elements: a list of final decisions (str) and a list of reasoning steps (list of tuples).
    """
    results = []
    final_decisions = []  # Collect final decisions for all graphs
    
    # Load question-specific data
    qtype_data = question_types.get(QType)
    if not qtype_data:
        raise ValueError(f"Unsupported question type: {QType}")
    
    for edge in graph_edges_list:
        # Call cumulative reasoning flow with edges, node count, and QType
        propositions_list, final_decision = cumulative_reasoning_flow(edge, n, hypothesis, QType, **kwargs)
        
        if propositions_list:
            if final_decision:
                # Append the results if both are valid
                results.append((propositions_list, final_decision))
                final_decisions.append(final_decision)
            else:
                print(f"Error: Missing final_decision for graph with edges {edge}")
        else:
            print(f"Error: Missing propositions_list for graph with edges {edge}")
    
    return final_decisions, results

def determine_correctness_by_graph_index(graph_index, final_decision, total_graphs):

    """
    Determines the correctness of the model's final decision based on the graph index.

    For the first half of the graphs, the correct answer is "No" (no cycle).
    For the second half of the graphs, the correct answer is "Yes" (cycle exists).

    Parameters:
        graph_index (int): The index of the current graph.
        final_decision (str): The model's final decision for the graph, either "yes" or "no".
        total_graphs (int): The total number of graphs being evaluated.

    Returns:
        tuple: A tuple containing a boolean indicating if the model's decision is correct,
               and the expected answer ("yes" or "no").
    """
    half_graphs = total_graphs // 2
    
    if graph_index < half_graphs:
        # First half: the correct answer is "No" (no cycle)
        expected_answer = "no"
    else:
        # Second half: the correct answer is "Yes" (cycle exists)
        expected_answer = "yes"
    
    # Debugging print statements
    # print(f"Debug Info - Graph Index: {graph_index}")
    # print(f"Debug Info - Total Graphs: {total_graphs}")
    # print(f"Debug Info - Half Graphs: {half_graphs}")
    # print(f"Debug Info - Final Decision: {final_decision.lower()}")
    # print(f"Debug Info - Expected Answer: {expected_answer}")

    # Compare the model's decision with the expected answer
    is_correct = final_decision.lower() == expected_answer.lower()
    # print(f"Debug Info - Is Correct: {is_correct}")

    # Compare the model's decision with the expected answer
    return is_correct, expected_answer

# Function to sort and group edges by their starting node
def sort_edges_by_node(edges):
    """
    Sorts the edges of an undirected graph by their starting node and returns a list of sorted edges.

    This function takes a list of edges, represented as tuples (u, v), and organizes them
    into a dictionary where each key is a node and the value is a list of nodes it is connected to.
    Since the graph is undirected, both (u, v) and (v, u) are considered. It then sorts the nodes
    and their connections, ensuring that each edge is added only once to the output list.

    Parameters:
        edges (list of tuples): A list of edges, where each edge is a tuple (u, v) representing
                                a connection between nodes u and v.

    Returns:
        list of lists: A list of sorted edges, where each edge is a list [u, v] and u < v.
    """
    edge_dict = {}  # Dictionary to group edges by their starting node
    for u, v in edges:
        if u not in edge_dict:
            edge_dict[u] = []
        edge_dict[u].append(v)

        # Since the graph is undirected, add the reverse edge too
        if v not in edge_dict:
            edge_dict[v] = []
        edge_dict[v].append(u)

    # Sort each node's connections and return a list of sorted edges
    sorted_edges = []
    for node in sorted(edge_dict):
        for connected_node in sorted(edge_dict[node]):
            if [node, connected_node] not in sorted_edges and [connected_node, node] not in sorted_edges:
                sorted_edges.append([node, connected_node])

    return sorted_edges

###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
###################################################################################################################################################################################################################################################################################################################################################################
# Section 6: Main functions.

def main():
    """
    The Main.
    """
    args = parse_arguments()
    QType = args.qtype  # Set the question type; adjust if needed for other types

    # Create a directory for logs if it doesn't exist
    log_dir = f'CR_logs_{QType}'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create a time-sensitive log file name within the logs directory
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_name = os.path.join(log_dir, f"CR prompting in ChatGPT 3.5, mode_{args.mode}, runtime_{current_time}.txt")

    # Set up logging configuration
    logging.basicConfig(
        filename=log_file_name,  # Dynamic log file name in the logs directory
        encoding='utf-8',
        level=logging.INFO,      # Log level
        format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
        datefmt='%Y-%m-%d %H:%M:%S'  # Date format
    )

    # Initialize result lists and counters
    results = []
    correct_predictions = 0

    # Get the total number of graphs based on difficulty mode
    graph_counts = {
        "easy": 150,   # Number of graphs for 'easy' mode
        "medium": 600, # Number of graphs for 'medium' mode
        "hard": 400    # Number of graphs for 'hard' mode
    }
    graph_count = graph_counts[args.mode]

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the graph files relative to the script directory
    graph_dir = os.path.join(script_dir, '..', 'NLGraph', QType, 'graph', args.mode, 'standard')

    try:
        for graph_index in range(graph_count):
            # Construct the full path to the graph file
            graph_file_path = os.path.join(graph_dir, f'graph{graph_index}.txt')

            # Check if the file exists
            if not os.path.isfile(graph_file_path):
                print(f"Graph file {graph_file_path} does not exist. Skipping.")
                continue

            with open(graph_file_path, "r") as graph_file:
                print(f"Processing graph {graph_index}...")

                # Read the first line for node count and edge count
                while True:
                    first_line = next(graph_file).strip()
                    if first_line:
                        break  # Found a non-empty line
                node_count_and_edges = first_line.split()
                if len(node_count_and_edges) < 2:
                    print(f"Invalid header in graph file {graph_index}. Skipping.")
                    continue
                node_count = int(node_count_and_edges[0])
                edge_count = int(node_count_and_edges[1])

                # Read the edges
                edges = []
                for line in graph_file:
                    line = line.strip()
                    if not line:
                        continue  # Skip empty lines
                    edge_nodes = line.split()
                    if len(edge_nodes) < 2:
                        print(f"Invalid edge data in graph {graph_index}: '{line}'. Skipping this edge.")
                        continue
                    u, v = map(int, edge_nodes[:2])
                    edges.append([u, v])
                print(f"Graph {graph_index} edges: {edges}")

            # Initialize variables based on QType
            kwargs = {}
            qtype_data = question_types.get(QType)
            if not qtype_data:
                raise ValueError(f"Unsupported question type: {QType}")

            if QType == 'connectivity':
                # Define node_a and node_b for each graph
                node_a = 0  # Replace with actual value or get from data
                node_b = node_count - 1  # Replace with actual value or get from data
                kwargs['node_a'] = node_a
                kwargs['node_b'] = node_b
                hypothesis = f"Is there a path between node {node_a} and node {node_b}?"
            elif QType == 'flow':
                source_node = 0  # Replace with actual value
                sink_node = node_count - 1  # Replace with actual value
                kwargs['source_node'] = source_node
                kwargs['sink_node'] = sink_node
                hypothesis = f"What is the maximum flow from node {source_node} to node {sink_node}?"
            elif QType == 'gnn':
                num_layers = 2  # Adjust as needed
                kwargs['num_layers'] = num_layers
                hypothesis = f"What are the embeddings of each node after {num_layers} layers of a simple graph convolution layer?"
            elif QType == 'hamiltonian':
                hypothesis = f"Is there a path in this graph that visits every node exactly once? If yes, give the path."
                # No additional kwargs needed
            elif QType == 'topology':
                hypothesis = f"Can all the nodes be visited? Give the solution."
                # No additional kwargs needed
            else:
                hypothesis = f"Determine whether there is a {qtype_data['property']} in the graph."
                # No additional kwargs needed

            # Call predict function
            final_decisions, _ = predict([edges], node_count, QType, hypothesis, **kwargs)
            final_decision = final_decisions[0] if final_decisions else None
            results.append(final_decision)

            # Handle None final_decision
            if final_decision is None:
                print(f"Warning: final_decision is None for graph {graph_index}.")
                final_decision = 'Unknown'

            # Determine correctness by comparing with expected answer
            is_correct, expected_answer = determine_correctness_by_graph_index(graph_index, final_decision, graph_count)
            if is_correct:
                correct_predictions += 1

            # Print the expected output
            print(f"Expected Answer: {expected_answer}")

            # Print the current score
            print(f"Score after graph {graph_index}: {correct_predictions} correct out of {graph_index + 1} graphs")
            print("=" * 90)

    except Exception as error:
        traceback.print_exc()
        print(f"An error occurred: {str(error)}")

    finally:
        if results:
            print(f"Final Results: {correct_predictions} correct out of {len(results)} graphs")

if __name__ == "__main__":
    main()

