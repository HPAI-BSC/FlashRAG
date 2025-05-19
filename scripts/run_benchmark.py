from flashrag.utils import get_dataset
from flashrag.pipeline import SequentialPipeline, LoglikelihoodPipeline
from flashrag.prompt import PromptTemplate
from flashrag.config import Config


import sys

def main(config_file):
  
  my_config = Config(config_file_path = config_file)

  all_split = get_dataset(my_config)
  test_data = all_split['test']

  prompt_template = PromptTemplate(
      my_config,
      system_prompt = "Answer the question based on the given documents.\n\n{reference}",
      #system_prompt="",
      user_prompt = "Question: {question}\nChoices:\n{choices}\nAnswer:",
      enable_chat=False,
  )
  pipeline = LoglikelihoodPipeline(
    my_config,
    prompt_template = prompt_template,
    add_choice_text=False
  )

  pipeline.run(test_data, do_eval=True)
  

if __name__ == "__main__":
  config_file = sys.argv[1]
  main(config_file)