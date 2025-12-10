# Graphusion: Your Personal Local Knowledge Graph Builder ⚙
**Graphusion: A RAG Framework for Scientific Knowledge Graph Construction with a Global Perspective**

Rui Yang, Boming Yang, Xinjie Zhao, Fan Gao, Aosong Feng, Sixun Ouyang, Moritz Blum, Tianwei She, Yuang Jiang, Freddy Lecue, Jinghui Lu, Irene Li;

Accepted by [NLP4KGC](https://sites.google.com/view/4rthnlp4kgc/home?authuser=0) workshop, WWW 2025. 

Graphusion is a pipeline that extracts Knowledge Graph triples from text.

![Architecture](graphusion_main.png)


## Setup
Create a new conda environment and install the required packages:
```
conda create -n graphusion python=3.10
conda activate graphusion
pip install -r requirements.txt
```

## Credentials
Sensitive data like API and database keys shall be stored in a `private_config.json` file in the root directory. 
The file should have the following structure:

```
{
  "OPENAI_API_KEY": "[key]",
  "GOOGLE_API_KEY": "[key]",
  "NEO4J": {
    "URI": "[uri]",
    "USER": "[user]",
    "PASSWORD":"[password]"
  }
}
```


## Usage
The pipeline processes text files from the `data/[dataset_name]/raw` directory (e.g., `data/test/raw`) as input. 
Furthermore, the pipeline requires relation definitions as a JSON file. This file defines the relations and 
provides a description of the relation (e.g., `data/test/relation_types.json`). In addition, some information can be 
provided to improve the results (`--gold_concept_file`, `--refined_concepts_file`, `--annotated_graph_file`)
or to skip pipeline steps (`--input_json_file`, `--input_triple_file`). See parameters below.

The ACL data is originally in a csv format. Therefore, we provide the notebook `preprocess.ipynb` to convert the 
data into the required text files.

The pipeline can be run using the following command:

```
usage: main.py [-h] [--run_name RUN_NAME] --dataset DATASET --relation_definitions_file RELATION_DEFINITIONS_FILE [--input_json_file INPUT_JSON_FILE]
               [--input_triple_file INPUT_TRIPLE_FILE] [--model MODEL] [--max_resp_tok MAX_RESP_TOK] [--max_input_char MAX_INPUT_CHAR]
               [--prompt_tpextraction PROMPT_TPEXTRACTION] [--prompt_fusion PROMPT_FUSION] [--gold_concept_file GOLD_CONCEPT_FILE]
               [--refined_concepts_file REFINED_CONCEPTS_FILE] [--annotated_graph_file ANNOTATED_GRAPH_FILE] [--language LANGUAGE] [--verbose]
               [--sample_size SAMPLE_SIZE]

options:
  -h, --help            show this help message and exit
  --run_name RUN_NAME   Assign a name to this run. The name will be used to, e.g., determine the output directory. We recommend to use unique and descriptive names to
                        distinguish the results of different models.
  --dataset DATASET     Name of the dataset. Is used to, e.g., determine the input directory.
  --relation_definitions_file RELATION_DEFINITIONS_FILE
                        Path to the relation definitions file. The file should be a JSON file, where the keys are the relation types and the values are dictionaries with the
                        following keys: 'label', 'description'.
  --input_json_file INPUT_JSON_FILE
                        Path to the input file. Step 1 will be skipped if this argument is provided. The input file should be a JSON file with the following structure:
                        {'concept1': [{'abstract': ['abstract1', ...], 'label: 0},...} E.g. data/test/concept_abstracts.json is the associated file createddurin step 1 in the
                        test run.
  --input_triple_file INPUT_TRIPLE_FILE
                        Path to the input file storing the triples in the format as outputted by the candidate triple extraction model. Step 1 and step 2 will be skipped if
                        this argument is provided.
  --model MODEL         Name of the LLM that should be used for the KG construction.
  --max_resp_tok MAX_RESP_TOK
                        Maximum number of tokens in the response of the candidate triple extraction model.
  --max_input_char MAX_INPUT_CHAR
                        Maximum number of characters in the input of the candidate triple extraction model.
  --prompt_tpextraction PROMPT_TPEXTRACTION
                        Path to the prompt template for step 1.
  --prompt_fusion PROMPT_FUSION
                        Path to the prompt template for fusion.
  --sample_size SAMPLE_SIZE
                        Limit processing to N items per step for testing with limited resources. 0 = no limit (process all). Recommended: 10-20 for quick tests.
  --gold_concept_file GOLD_CONCEPT_FILE
                        Path to a file with concepts that are provided by experts. The file should be a tsv file, each row should look like: 'concept id | concept
  --refined_concepts_file REFINED_CONCEPTS_FILE
                        In step 2 (candidate triple extraction) many new concepts might be added. Instead of using these, concepts can be provided through this parameter.
                        Specify the path to a file with refined concepts of the graph. The file should be a tsv file, each row should look like: "concept id | concept name"
  --annotated_graph_file ANNOTATED_GRAPH_FILE
                        Path to the annotated graph.
  --language LANGUAGE   Language of the abstracts.
  --verbose             Print additional information to the console.
```

## LLM Providers
Graphusion supports OpenAI, Google Gemini, and Ollama (local) models. The CLI argument `--model` determines which
backend is used:

- Model names starting with `gemini` use the Google Gemini API and require `GOOGLE_API_KEY`.
- Model names starting with `ollama:` use a local Ollama server (e.g., `ollama:llama3.2`, `ollama:mistral`).
- All other model names default to the OpenAI API and require `OPENAI_API_KEY`.

Set the respective key either as an environment variable or inside `private_config.json`. You only need to
provide the key for the backend you plan to use.

### Using Ollama (Local Models)
To use Ollama for completely local inference without any API keys:

1. Install Ollama from https://ollama.ai
2. Pull a model: `ollama pull llama3.2` (or `mistral`, `qwen2.5`, etc.)
3. Run the pipeline with the `ollama:` prefix:
   ```
   python main.py --model "ollama:llama3.2" --run_name "test" --dataset "test" ...
   ```

By default, Ollama is expected at `http://localhost:11434`. You can override this by:
- Setting the `OLLAMA_BASE_URL` environment variable
- Adding `"OLLAMA_BASE_URL": "http://your-host:port"` to `private_config.json`

The output of the pipeline are the following files: 
- `concept_abstracts`: The json file mapping the extracted concepts to their abstracts.
- `step-02.jsonl`: The extracted triples in linewise JSON format.
- `step-03.jsonl`: The fused triples in linewise JSON format.


## Example 
To run the full pipeline on a small sample (`test`) dataset, call: 
`python main.py --run_name "test" --dataset "test" --relation_definitions_file "data/test/relation_types.json" --gold_concept_file "data/test/gold_concepts.tsv" --refined_concepts_file "data/test/refined_concepts.tsv"`

To reproduce the Graphusion results on the ACL (`nlp) dataset, call:
`python main.py --run_name "acl" --dataset "nlp" --relation_definitions_file "data/nlp/relation_types.json" --gold_concept_file "data/nlp/gold_concepts.tsv" --refined_concepts_file "data/nlp/refined_concepts.tsv"`

### Quick Test with Limited Resources
If you have limited computational resources (e.g., small GPU, free Colab tier), use `--sample_size` to process only a subset of the data:

```bash
# Process only 10 concepts (quick test ~2-5 minutes)
python main.py --run_name "quick_test" --dataset "test" \
    --relation_definitions_file "data/test/relation_types.json" \
    --model "ollama:llama3.2:1b" \
    --sample_size 10

# Process 20 concepts (more thorough test ~5-10 minutes)
python main.py --run_name "sample_test" --dataset "test" \
    --relation_definitions_file "data/test/relation_types.json" \
    --model "gemini-1.5-flash" \
    --sample_size 20
```

This limits:
- **Step 1**: Raw texts to `sample_size * 10` (to extract enough concepts)
- **Step 2**: Concepts to process to `sample_size`
- **Step 3**: Concepts to fuse to `sample_size`

## Credits
This implementation is based on the code of Rui Yang and Irene Li. Moritz Blum extended their code and implemented this pipeline.


## Cite
```
@inproceedings{yang2025graphusion,
  title={Graphusion: A RAG Framework for Scientific Knowledge Graph Construction with a Global Perspective},
  author={Yang, Rui and Yang, Boming and Zhao, Xinjie and Gao, Fan and Feng, Aosong and Ouyang, Sixun and Blum, Moritz and She, Tianwei and Jiang, Yuang and Lecue, Freddy and Lu, Jinghui and Li, Irene},
  booktitle={Proceedings of the NLP4KGC Workshop at the World Wide Web Conference (WWW) 2025},
  year={2025}
}
```
