import json
import pandas as pd
import random
import os
from langchain_core.prompts import ChatPromptTemplate
from tqdm import tqdm

from graphs import get_nx_graph, verbalize_neighbors_triples_from_triples, \
    verbalize_neighbors_triples_from_graph


def step_03_fusion(model: any,
                   input_file: str,
                   output_file: str,
                   relation_def: dict[str, dict[str, str]],
                   relation_2_id: dict[str, int],
                   data: dict[str, dict[str, list[str]]],
                   logging: any,
                   config: dict[str, any]):
    """
    Step 2: Fusion
    Refines the KG by fusing the candidate triples with the prerequisite-of graph or makes i
    t more self-consistent.

    :param model: the language model to use
    :param input_file: the file with the candidate triples
    :param output_file: the file to write the refined triples to
    :param relation_def: the relation definitions
    :param relation_2_id: the relation to id mapping
    :param data: the abstracts per concept
    :param logging: the logger
    :param config: the configuration can be provided with the following keys: refined_concepts_file,
    annotated_graph_file, prompt_fusion, max_input_char
    :return: None
    """

    logging.info("Step 2: Starting global fusion to refine the KG.")

    if 'refined_concepts_file' not in config:
        config['refined_concepts_file'] = None
        logging.info(f"No refined concepts file provided. Proceeding without it.")
    if 'annotated_graph_file' not in config:
        config['annotated_graph_file'] = ""
        logging.info(f"No annotated graph file provided. Proceeding without it.")
    if 'prompt_fusion' not in config or not config['prompt_fusion']:
        config['prompt_fusion'] = "prompts/prompt_fusion.txt"
        logging.info(f"No prompt template for fusion provided. Using default prompt: "
                     f"{config['prompt_fusion']}")
    
    # Verify prompt file exists
    if not os.path.exists(config['prompt_fusion']):
        raise FileNotFoundError(f"Prompt file not found: {config['prompt_fusion']}")

    candidate_triples = []
    for line in open(input_file, 'r'):
        t = json.loads(line)
        candidate_triples.append((t['s'], t['p'], t['o']))

    if config['refined_concepts_file'] is not None and config['refined_concepts_file'] != "":
        logging.info(
            f"Refined concepts specified. Loading concepts from {config['refined_concepts_file']}.")
        id_2_concept = {i: str(c['concept']) for i, c in
                        pd.read_csv(config['refined_concepts_file'], sep='|', header=None,
                                    names=['id', 'concept'], index_col=0).iterrows()}
        logging.info(
            f"Loaded {len(id_2_concept)} refined concepts, e.g. {', '.join(list(id_2_concept.values())[:3])}")
    else:
        # randomly sample up to 100 concepts extracted in step 2
        concepts = [c[0] for c in candidate_triples] + [c[2] for c in candidate_triples]
        random.shuffle(concepts)
        logging.info(
            f'No refined concepts specified. Randomly selected concepts: {", ".join(concepts[:100])}')
        id_2_concept = {i: c for i, c in enumerate(concepts)}

    # Apply sampling if specified
    sample_size = config.get('sample_size', 0)
    if sample_size > 0:
        original_size = len(id_2_concept)
        sampled_items = list(id_2_concept.items())[:sample_size]
        id_2_concept = dict(sampled_items)
        logging.info(f"SAMPLING MODE: Reduced concepts from {original_size} to {len(id_2_concept)} (--sample_size={sample_size})")

    concept_2_id = {v: k for k, v in id_2_concept.items()}

    # build the prerequisite-of graph
    prerequisite_of_triples = []
    if config['annotated_graph_file'] and config['annotated_graph_file'] != "" and os.path.exists(config['annotated_graph_file']):
        logging.info(f"Loading annotated graph from {config['annotated_graph_file']}.")
        with open(config['annotated_graph_file'], 'r') as f:
            for line in f:
                s, p, o = line.strip().split('\t')
                prerequisite_of_triples.append((str(s), str(p), str(o)))
    else:
        logging.info(
            f"No annotated graph provided or file not found. Proceeding without it.")

    prerequisite_of_graph = get_nx_graph(prerequisite_of_triples, concept_2_id, relation_2_id)

    # initialize the prompt template
    prompt_template_txt = open(config['prompt_fusion']).read()

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a knowledge graph builder."),
        ("user", prompt_template_txt)
    ])

    output_stream = open(output_file, 'w')
    for id, candidate_concept in tqdm(id_2_concept.items(), total=len(id_2_concept)):
        candidate_subgraph = verbalize_neighbors_triples_from_triples(candidate_triples,
                                                                      candidate_concept)

        prerequisite_of_graph_subgraph = verbalize_neighbors_triples_from_graph(
            prerequisite_of_graph, candidate_concept, concept_2_id, id_2_concept, mode='outgoing')
        abstracts = ' '.join(
            data[candidate_concept]['abstracts']) if candidate_concept in data else ''

        prompt = prompt_template.invoke(
            {"concept": candidate_concept,
             "graph1": candidate_subgraph,
             "graph2": prerequisite_of_graph_subgraph,
             "background": abstracts[:config['max_input_char']],
             "relation_definitions": '\n'.join(
                 [f"{rel_type}: {rel_data['description']}" for rel_type, rel_data in
                  relation_def.items()])})

        # query the model
        response = model.invoke(prompt)

        if response != "None" and response:
            try:
                response_json = json.loads(response)
                
                # Handle case where response is not a list
                if not isinstance(response_json, list):
                    logging.warning(f"Concept '{candidate_concept}': Expected list, got {type(response_json).__name__}. Skipping.")
                    continue
                
                for triple in response_json:
                    # Validate triple is a dictionary with required keys
                    if not isinstance(triple, dict):
                        logging.warning(f"Concept '{candidate_concept}': Triple is not a dict, got {type(triple).__name__}. Skipping.")
                        continue
                    if 'p' not in triple or 's' not in triple or 'o' not in triple:
                        logging.warning(f"Concept '{candidate_concept}': Triple missing required keys (s, p, o). Skipping: {triple}")
                        continue
                    
                    if triple['p'] not in list(relation_2_id.keys()):
                        continue
                    output_stream.write(json.dumps(triple) + '\n')
            except json.JSONDecodeError as e:
                logging.warning(f"Concept '{candidate_concept}': Failed to parse JSON response: {e}. Response: {response[:200]}...")
                continue
    output_stream.close()
    logging.info("Step 2: Fusion completed.")
